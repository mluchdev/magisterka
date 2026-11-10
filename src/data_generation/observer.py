import csv
import json
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/opt/cuda")  # vLLM/FlashInfer compile kernels with nvcc
os.environ["PATH"] = f"{Path(sys.executable).parent}:/opt/cuda/bin:{os.environ['PATH']}"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
csv.field_size_limit(sys.maxsize)

import numpy as np
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from common import DATA, OBSERVER_MODEL
from data_generation.openturingbench import load_otb

OUT = DATA / "observer"
TOP_K, MAX_TEXT_TOKENS, MAX_SUMMARY_TOKENS, SAVE_EVERY_S, SEED = 20, 5000, 400, 300, 0
# GPU memory (8 GB): 0.74 is a floor for the KV cache of a 6144-token sequence, 0.80+ runs out of memory for the logits
GPU_UTIL, BATCHED_TOKENS, MAX_SEQS, CHUNK_TEXTS = 0.74, 256, 4, 8
NO_THINK = {"enable_thinking": False}
SUMMARIZE = ("Write a detailed summary (120-180 words) of the following article. Include the key facts: who, what, where, when, "
             "names, organizations, places, numbers and dates. Do not copy sentences. Do not add any names, numbers, dates or "
             "facts that are not in the article. Reply with the summary only.\n\nArticle:\n{}")
WRAP = "Write a news article based on this summary.\n\nSummary: {}"  # the scoring context
HEADER = ["text_id", "url", "model", "label", "row_start", "row_end", "truncated", "prompt", "text"]


class Observer:
    """Nemotron (vLLM). Per article: a summary (SUMMARIZE, greedy), then the article teacher-forced after a fresh
    context holding only that summary (WRAP); per token 21 floats = top-20 next-token p (descending) + p(actual token)."""

    def __init__(self):
        # max_num_batched_tokens: the logits of a chunk (x 131k vocab x 4 B) must fit next to the weights on the 8 GB GPU
        self.llm = LLM(OBSERVER_MODEL, max_model_len=6144, gpu_memory_utilization=GPU_UTIL, max_num_seqs=MAX_SEQS,
                       max_num_batched_tokens=BATCHED_TOKENS, trust_remote_code=True, mamba_ssm_cache_dtype="float32",
                       kv_cache_dtype="fp8", enforce_eager=True, logprobs_mode="raw_logprobs")
        self.tok = self.llm.get_tokenizer()

    def cut(self, text):
        """(text cut to MAX_TEXT_TOKENS tokens, its token ids, whether it was cut)."""
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= MAX_TEXT_TOKENS:
            return text, ids, False
        return self.tok.decode(ids[:MAX_TEXT_TOKENS]), ids[:MAX_TEXT_TOKENS], True

    def summarize(self, texts):
        chats = [[{"role": "user", "content": SUMMARIZE.format(t)}] for t in texts]
        outs = self.llm.chat(chats, SamplingParams(temperature=0, max_tokens=MAX_SUMMARY_TOKENS), chat_template_kwargs=NO_THINK)
        return [o.outputs[0].text.strip() for o in outs]

    @staticmethod
    def token_row(logprobs, token_id):
        top = sorted((v for v in logprobs.values() if v.rank <= TOP_K), key=lambda v: v.rank)  # drops the actual token if it ranks lower
        row = np.zeros(TOP_K + 1, dtype=np.float32)
        row[: len(top)] = np.exp([v.logprob for v in top])
        row[TOP_K] = np.exp(logprobs[token_id].logprob)
        return row

    def score(self, summaries, ids_list):
        """Per article an array (n_tokens, 21)."""
        prefixes = [self.tok(self.tok.apply_chat_template([{"role": "user", "content": WRAP.format(s)}], tokenize=False,
                                                          add_generation_prompt=True, **NO_THINK), add_special_tokens=False)["input_ids"]
                    for s in summaries]
        outs = self.llm.generate([{"prompt_token_ids": p + i} for p, i in zip(prefixes, ids_list)],
                                 SamplingParams(max_tokens=1, prompt_logprobs=TOP_K))
        return [np.stack([self.token_row(o.prompt_logprobs[len(p) + k], t) for k, t in enumerate(i)])
                for o, p, i in zip(outs, prefixes, ids_list)]

    def __call__(self, texts):
        """[(summary, array (n_tokens, 21), truncated)] per text."""
        cuts = [self.cut(t) for t in texts]
        summaries = self.summarize([t for t, _, _ in cuts])
        return list(zip(summaries, self.score(summaries, [i for _, i, _ in cuts]), [c for _, _, c in cuts]))


@dataclass
class ObserverStore:
    """data/observer/<split>.npy (float32 (n_tokens, 21)) + <split>_index.csv (HEADER; rows [row_start, row_end) of the
    .npy) + <split>_meta.json (settings; a resume must match them). Resumes from existing files; saves atomically."""

    split: str
    META = {"model": OBSERVER_MODEL, "variant": "summary", "seed": SEED, "top_k": TOP_K, "max_text_tokens": MAX_TEXT_TOKENS, "labels_cut": True}

    def __post_init__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        self.npy, self.csv, self.meta = (OUT / f"{self.split}{x}" for x in (".npy", "_index.csv", "_meta.json"))
        self.rows, self.index = [], []
        if self.meta.exists():
            assert json.loads(self.meta.read_text()) == self.META, f"{self.meta} was made with other settings, delete it or fix them"
            self.index = list(csv.reader(open(self.csv)))[1:]
            self.rows = [np.load(self.npy)[: self.n_rows]] if self.index else []  # drop rows of a save interrupted between the files
            print(f"resuming: {len(self.index)} texts done", flush=True)

    @property
    def n_rows(self):
        return int(self.index[-1][HEADER.index("row_end")]) if self.index else 0

    def add(self, arr, **fields):
        """fields: every HEADER column except row_start/row_end."""
        fields |= {"row_start": self.n_rows, "row_end": self.n_rows + len(arr)}
        self.index.append([fields[h] for h in HEADER])
        self.rows.append(arr)

    def save(self):  # atomic: a Ctrl+C during saving cannot corrupt the files
        for path, mode, write in ((self.npy, "wb", lambda f: np.save(f, np.concatenate(self.rows))),
                                  (self.csv, "w", lambda f: csv.writer(f).writerows([HEADER, *self.index]))):
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, mode, newline=None if mode == "wb" else "") as f:
                write(f)
            os.replace(tmp, path)
        self.meta.write_text(json.dumps(self.META))


def run(name, rows, observer):
    """Score rows = [(text_id, url, text, label, model)] into data/observer/<name>.* (resumes after the saved texts)."""
    store, last_save, interrupted = ObserverStore(name), time.time(), []
    done = [r[HEADER.index("text")] for r in store.index]
    assert done == [r[2] for r in rows[:len(done)]], f"{store.csv} holds other texts than the input: delete data/observer/{name}.npy, _index.csv, _meta.json"

    def on_sigint(*_):  # vLLM aborts the running requests on Ctrl+C and may return broken outputs instead of raising
        interrupted.append(1)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_sigint)
    try:
        for c in range(len(store.index), len(rows), CHUNK_TEXTS):
            chunk = rows[c:c + CHUNK_TEXTS]
            for (text_id, url, text, label, model), (summary, arr, trunc) in zip(chunk, observer([r[2] for r in chunk])):
                store.add(arr, text_id=text_id, url=url, model=model, label=label, truncated=trunc, prompt=summary, text=text)
            print(f"{name}: {len(store.index)}/{len(rows)} texts, {store.n_rows} tokens", flush=True)
            if time.time() - last_save > SAVE_EVERY_S:
                store.save()
                last_save = time.time()
    except BaseException:
        if not interrupted:  # a real error: the finally below still saves, then it is raised
            raise
        print("interrupted, saving (the unfinished chunk of texts is dropped)", flush=True)
    finally:
        if store.rows:
            store.save()
            print("saved", store.npy, flush=True)


def main(split="test", n_texts=None):
    items = load_otb(split)
    order = random.Random(SEED).sample(range(len(items)), len(items))[:n_texts]  # fixed order: any prefix is a random sample
    run(split, [(f"{split}_{n}", *items[o]) for n, o in enumerate(order)], Observer())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test", int(sys.argv[2]) if len(sys.argv) > 2 else None)
