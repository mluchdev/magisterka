"""Observer: Nemotron reads whole OpenTuringBench articles and we save its next-token probabilities.

Per article: (1) Nemotron writes a detailed summary of it (key people, places, dates, numbers, events), so that the facts
are known and do not surprise it later; (2) in a fresh context that holds ONLY that summary (wrapped in WRAP), the whole
article is fed as the answer (teacher forcing) and for every token we record the 20 highest next-token probabilities
(descending) + the probability of the actual token = 21 numbers. Probabilities are exp(log-prob) from the model's
full-vocabulary distribution (no renormalisation, so the top 20 sum to < 1).

Files in data/observer/ (one growing pair per split, resumable):
  <split>.npy         float32 (n_tokens, 21); rows [row_start, row_end) of an article are in the index
  <split>_index.csv   text_id (test_<n>, n = position in the fixed shuffled order of the split's texts), url, model, label,
                      row_start, row_end, truncated, prompt (the summary), text
  <split>_meta.json   model, prompt variant, seed (a resume must match them)
Ctrl+C saves and exits; running again continues after the last finished chunk of texts.
Run (needs vLLM, so the vllm venv): .venv-vllm/bin/python src/data_generation/observer.py [split] [n_texts]
"""

import csv
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/opt/cuda")  # vLLM/FlashInfer compile kernels with nvcc
os.environ["PATH"] = f"{Path(sys.executable).parent}:/opt/cuda/bin:{os.environ['PATH']}"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
csv.field_size_limit(sys.maxsize)

import numpy as np
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from data_generation.openturingbench import OTBDataset

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"
OUT = Path(__file__).parent.parent.parent / "data" / "observer"
TOP_K, MAX_TEXT_TOKENS, MAX_SUMMARY_TOKENS, SAVE_EVERY_S, SEED = 20, 5000, 400, 300, 0
# memory settings, overridable by run_observer.py, which lowers them after an out-of-memory crash
GPU_UTIL = float(os.environ.get("OBS_GPU_UTIL", 0.74))
BATCHED_TOKENS = int(os.environ.get("OBS_BATCHED_TOKENS", 256))
MAX_SEQS = int(os.environ.get("OBS_MAX_SEQS", 4))
CHUNK_TEXTS = int(os.environ.get("OBS_CHUNK", 8))
NO_THINK = {"enable_thinking": False}
CONTROL = "Write a news article."  # variant "none": the same generic prompt for every article
RULES = "Do not copy sentences. Do not add any names, numbers, dates or facts that are not in the article. Reply with the summary only."
PROMPTS = {
    "summary": "Write a detailed summary (120-180 words) of the following article. Include the key facts: who, what, where, "
               "when, names, organizations, places, numbers and dates. " + RULES + "\n\nArticle:\n{}",
    "brief": "Summarize the following article in at most 30 words. " + RULES + "\n\nArticle:\n{}",
}
VARIANT = "summary"
WRAP = "Write a news article based on this summary.\n\nSummary: {}"


def load_llm():
    # max_num_batched_tokens: the logits of a chunk (x 131k vocab x 4 B) must fit next to the weights on the 8 GB GPU
    llm = LLM(MODEL_ID, max_model_len=6144, gpu_memory_utilization=GPU_UTIL, max_num_seqs=MAX_SEQS, max_num_batched_tokens=BATCHED_TOKENS,
              trust_remote_code=True, mamba_ssm_cache_dtype="float32", kv_cache_dtype="fp8", enforce_eager=True,
              logprobs_mode="raw_logprobs")
    return llm, llm.get_tokenizer()


def cut(tok, text):
    """(text cut to MAX_TEXT_TOKENS tokens, its token ids, whether it was cut)."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= MAX_TEXT_TOKENS:
        return text, ids, False
    return tok.decode(ids[:MAX_TEXT_TOKENS]), ids[:MAX_TEXT_TOKENS], True


def ask(llm, texts, variant=VARIANT):
    """One summary per article (greedy)."""
    if variant == "none":
        return [CONTROL] * len(texts)
    chats = [[{"role": "user", "content": PROMPTS[variant].format(t)}] for t in texts]
    outs = llm.chat(chats, SamplingParams(temperature=0, max_tokens=MAX_SUMMARY_TOKENS), chat_template_kwargs=NO_THINK)
    return [o.outputs[0].text.strip() for o in outs]


def token_row(logprobs, token_id):
    """21 floats: probabilities of the TOP_K most likely tokens (descending), then of the actual token."""
    top = sorted((v for v in logprobs.values() if v.rank <= TOP_K), key=lambda v: v.rank)  # drops the actual token if it ranks lower
    row = np.zeros(TOP_K + 1, dtype=np.float32)
    row[: len(top)] = np.exp([v.logprob for v in top])
    row[TOP_K] = np.exp(logprobs[token_id].logprob)
    return row


def score(llm, tok, summaries, ids_list, variant=VARIANT):
    """Per article an array (n_tokens, 21): the article is the answer to a context that holds only its summary."""
    batch = []
    for summary, ids in zip(summaries, ids_list):
        msg = summary if variant == "none" else WRAP.format(summary)
        prefix = tok(tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                     add_generation_prompt=True, **NO_THINK), add_special_tokens=False)["input_ids"]
        batch.append((prefix, ids))
    outs = llm.generate([{"prompt_token_ids": p + i} for p, i in batch],
                        SamplingParams(max_tokens=1, prompt_logprobs=TOP_K))
    return [np.stack([token_row(o.prompt_logprobs[len(p) + k], t) for k, t in enumerate(i)])
            for o, (p, i) in zip(outs, batch)]


def main(split="test", n_texts=None):
    OUT.mkdir(parents=True, exist_ok=True)
    npy, csv_path, meta_path = OUT / f"{split}.npy", OUT / f"{split}_index.csv", OUT / f"{split}_meta.json"
    meta = {"model": MODEL_ID, "variant": VARIANT, "seed": SEED, "top_k": TOP_K, "max_text_tokens": MAX_TEXT_TOKENS, "labels_cut": True}
    header = ["text_id", "url", "model", "label", "row_start", "row_end", "truncated", "prompt", "text"]

    items = OTBDataset(split).items
    order = random.Random(SEED).sample(range(len(items)), len(items))[:n_texts]  # fixed order: any prefix is a random sample
    rows, index = [], []
    if meta_path.exists():  # resume
        assert json.loads(meta_path.read_text()) == meta, f"{meta_path} was made with other settings, delete it or fix them"
        index = list(csv.reader(open(csv_path)))[1:]
        rows = [np.load(npy)[: int(index[-1][5])]] if index else []  # drop rows of a save interrupted between the files
        print(f"resuming: {len(index)} texts done", flush=True)
    n_rows = int(index[-1][5]) if index else 0

    def save():  # atomic: a Ctrl+C during saving cannot corrupt the files
        for path, write in ((npy, lambda f: np.save(f, np.concatenate(rows))), (csv_path, lambda f: csv.writer(f).writerows([header, *index]))):
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "wb" if path == npy else "w", newline=None if path == npy else "") as f:
                write(f)
            os.replace(tmp, path)
        meta_path.write_text(json.dumps(meta))

    llm, tok = load_llm()
    last_save, interrupted = time.time(), []

    def on_sigint(*_):  # vLLM aborts the running requests on Ctrl+C and may return broken outputs instead of raising
        interrupted.append(1)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_sigint)
    try:
        for c in range(len(index), len(order), CHUNK_TEXTS):
            chunk = [(c + j, *items[order[c + j]]) for j in range(min(CHUNK_TEXTS, len(order) - c))]  # (n, url, text, label, model)
            cuts = [cut(tok, text) for _, _, text, _, _ in chunk]
            summaries = ask(llm, [t for t, _, _ in cuts])
            arrs = score(llm, tok, summaries, [i for _, i, _ in cuts])
            for (n, url, text, label, model), (_, _, trunc), summary, arr in zip(chunk, cuts, summaries, arrs):
                index.append([f"{split}_{n}", url, model, label, n_rows, n_rows + len(arr), trunc, summary, text])
                rows.append(arr)
                n_rows += len(arr)
            print(f"{len(index)}/{len(order)} texts, {n_rows} tokens", flush=True)
            if time.time() - last_save > SAVE_EVERY_S:
                save()
                last_save = time.time()
    except BaseException:
        if not interrupted:  # a real error: the finally below still saves, then it is raised
            raise
        print("interrupted, saving (the unfinished chunk of texts is dropped)", flush=True)
    finally:
        if rows:
            save()
            print("saved", npy, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test", int(sys.argv[2]) if len(sys.argv) > 2 else None)
