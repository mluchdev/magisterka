import itertools
import json
import random
import re
import sys
from functools import cache
from pathlib import Path

import numpy as np
import regex
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from common import DATA
from data_generation.openturingbench import load_otb

# Ports of RAID's attacks (github.com/liamdugan/raid, generation/adversarial/attackers): same rates, RNG use and logic,
# except: insert_paragraphs keeps the original text, glue is ours, synonym uses ModernBERT instead of bert-base-cased.
# Every attack is a function (text, rate) -> (attacked text, number of edits); rate = fraction of the candidates to edit.
ATTACK_DIR = DATA / "attacks"
RESOURCE_DIR = ATTACK_DIR / "resources"
US_TO_UK_URL = "https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/american_spellings.json"
MISSPELLINGS_URL = "https://raw.githubusercontent.com/liamdugan/raid/main/generation/adversarial/attackers/resources/misspelling3.json"


@cache
def download_json(url):
    """JSON file from url, downloaded once into data/attacks/resources."""
    path = RESOURCE_DIR / url.split("/")[-1]
    if not path.exists():
        RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(requests.get(url, timeout=60).content)
    return json.loads(path.read_text())


@cache
def nltk_tools():
    """nltk module, sentence splitter (Punkt, as RAID), word tokenizer."""
    import nltk
    for resource in ("punkt_tab", "averaged_perceptron_tagger_eng", "stopwords"):
        nltk.download(resource, quiet=True)
    return nltk, nltk.tokenize.PunktTokenizer("english"), nltk.tokenize.NLTKWordTokenizer()


def split_words(text):
    """RAID's tokenisation: text -> sentences -> words.
    Returns word_spans = [(start, end) of each word in text] and
    pieces = [word0, gap0, word1, gap1, ..., last word] (gap = text between two words, e.g. " " or ", "),
    so words are at even indices and "".join(pieces) rebuilds the text (without anything before the first / after the last word)."""
    _, sentence_splitter, word_tokenizer = nltk_tools()
    word_spans = [(sent_start + word_start, sent_start + word_end)
                  for sent_start, sent_end in sentence_splitter.span_tokenize(text)
                  for word_start, word_end in word_tokenizer.span_tokenize(text[sent_start:sent_end])]
    boundaries = list(itertools.chain.from_iterable(word_spans))  # start0, end0, start1, end1, ...
    pieces = [text[start:end] for start, end in itertools.pairwise(boundaries)]
    return word_spans, pieces


@cache
def us_to_uk():
    """{american: british} spelling dict and one regex matching any of its keys."""
    spellings = download_json(US_TO_UK_URL)
    return spellings, regex.compile(r"\L<words>", words=spellings.keys())


def alternative_spelling(text, rate):
    """American -> British spelling for a fraction `rate` of the matches (substring matches, as RAID)."""
    spellings, pattern = us_to_uk()
    matches = list(pattern.finditer(text))
    chosen = random.sample(matches, int(len(matches) * rate))
    for match in sorted(chosen, key=lambda m: m.start(), reverse=True):  # from the end, so earlier positions stay valid
        text = text[:match.start()] + spellings[match.group()] + text[match.end():]
    return text, len(chosen)


def article_deletion(text, rate):
    """Delete a fraction `rate` of the lowercase articles " a ", " an ", " the " (the space after it stays)."""
    def positions_of(article):
        pos = text.find(article)
        while pos != -1:
            yield pos
            pos = text.find(article, pos + len(article) - 1)  # -1: the trailing space can start the next article
    found = [(pos, article) for article in (" a ", " an ", " the ") for pos in positions_of(article)]
    chosen = random.sample(found, int(len(found) * rate))
    for pos, article in sorted(chosen, reverse=True):
        text = text[:pos] + text[pos + len(article) - 1:]
    return text, len(chosen)


def insert_paragraphs(text, rate):
    """Replace a fraction `rate` of the gaps between sentences with a paragraph break "\\n\\n"."""
    sentences = list(nltk_tools()[1].span_tokenize(text))  # [(start, end)] of each sentence
    chosen = random.sample(range(1, len(sentences)), int((len(sentences) - 1) * rate))  # i = gap before sentence i
    for i in sorted(chosen, reverse=True):
        text = text[:sentences[i - 1][1]] + "\n\n" + text[sentences[i][0]:]
    return text, len(chosen)


def whitespace(text, rate):
    """Add an extra space after rate * (number of words) randomly chosen words (a word can be chosen twice)."""
    words = text.split(" ")
    chosen = random.choices(range(len(words)), k=int(len(words) * rate))
    for i in chosen:
        words[i] += " "
    return " ".join(words), len(chosen)


def glue(text, rate):
    """(Ours, not in RAID.) Delete a fraction `rate` of the single spaces between two word characters."""
    spaces = [m.start() for m in re.finditer(r"(?<=\w) (?=\w)", text)]
    chosen = random.sample(spaces, int(len(spaces) * rate))
    for pos in sorted(chosen, reverse=True):
        text = text[:pos] + text[pos + 1:]
    return text, len(chosen)


def number(text, rate):
    """Replace every digit of a fraction `rate` of the numbers (regex \\d+\\.?\\d*) with a random digit."""
    numbers = [m.span() for m in re.finditer(r"\d+\.?\d*", text)]
    chosen = sorted(random.sample(numbers, int(len(numbers) * rate)))
    chars = list(text)
    for start, end in chosen:
        for i in range(start, end):
            chars[i] = random.choice("0123456789") if chars[i].isnumeric() else chars[i]
    return "".join(chars), len(chosen)


@cache
def gpt2():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    return tokenizer, AutoModelForCausalLM.from_pretrained("openai-community/gpt2").cuda().eval(), torch


def gpt2_word_scores(text, candidates):
    """GPT-2 small log-probability of each candidate word = sum over the GPT-2 tokens overlapping it
    (GPT-2 sees only the first 1024 tokens, words after that get 0)."""
    tokenizer, model, torch = gpt2()
    ids = tokenizer(tokenizer.bos_token + text, truncation=True, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        logprobs = torch.log_softmax(model(ids).logits[0, :-1], -1)  # prediction of token t+1 at position t
        token_logprobs = logprobs.gather(1, ids[0, 1:, None])[:, 0].cpu().tolist()
    tokens = [(tokenizer.decode(t), lp) for t, lp in zip(ids[0, 1:].tolist(), token_logprobs) if t not in tokenizer.all_special_ids]
    token_ends = list(itertools.accumulate(len(token_text) for token_text, _ in tokens))  # char offsets in text
    token_spans = list(zip([0] + token_ends[:-1], token_ends))
    return [sum(tokens[j][1] for j, (tok_start, tok_end) in enumerate(token_spans) if word_start < tok_end and word_end > tok_start)
            for _, word_start, word_end in candidates]


def misspelling(text, rate):
    """Replace rate * (number of words) words that have a common misspelling (RAID's misspelling3.json:
    {word: [misspellings]}) with a random one of them. If there are more candidates than needed, the ones GPT-2
    finds most likely (most predictable) are taken."""
    misspellings = download_json(MISSPELLINGS_URL)
    word_spans, pieces = split_words(text)
    boundaries = itertools.pairwise(itertools.chain.from_iterable(word_spans))
    candidates = [(i, start, end) for i, (start, end) in enumerate(boundaries) if text[start:end].lower() in misspellings]
    n_change = min(int(len(word_spans) * rate), len(candidates))
    if n_change == 0:
        return text, 0
    if n_change < len(candidates):
        scores = gpt2_word_scores(text, candidates)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        candidates = [c for c, score in ranked if score != 0][:n_change]
    for i, start, end in sorted(candidates[:n_change]):  # i = index into pieces
        new_word = random.choice(misspellings[text[start:end].lower()])
        pieces[i] = new_word.capitalize() if text[start].isupper() else new_word
    return "".join(pieces), len(candidates[:n_change])


@cache
def synonym_models():
    import fasttext
    from transformers import pipeline
    nltk = nltk_tools()[0]
    fill_mask = pipeline("fill-mask", model="answerdotai/ModernBERT-base", device=0)
    word_vectors = fasttext.load_model(str(RESOURCE_DIR / "cc.en.300.bin"))
    return fill_mask, word_vectors, nltk.tag.PerceptronTagger(), set(nltk.corpus.stopwords.words("english"))


def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def synonym(text, rate, top_k=20, min_prob=0.0025, min_cos=0.5, bert_window=40, pos_window=8):
    """Swap rate * (number of words) words for a synonym. For every word (alphabetic, not a stopword):
    mask it (ModernBERT sees bert_window pieces = 20 words on each side) and take ModernBERT's top_k guesses; keep a
    guess if it is alphabetic, differs from the word, has the same POS tag (NLTK, pos_window pieces each side),
    fastText cosine to the word >= min_cos and ModernBERT probability >= min_prob; the most probable kept guess is the
    word's swap. Then the swaps with the highest probability are applied."""
    fill_mask, word_vectors, tagger, stopwords = synonym_models()
    word_spans, pieces = split_words(text)
    pos_tags = [tag for _, tag in tagger.tag([text[s:e] for s, e in word_spans])]  # one per word = per even piece
    targets = [(i, tag) for i, tag in zip(range(0, len(pieces), 2), pos_tags) if pieces[i].isalpha() and pieces[i].lower() not in stopwords]
    masked_texts = ["".join(pieces[max(0, i - bert_window):i] + [fill_mask.tokenizer.mask_token] + pieces[i + 1:i + bert_window])
                    for i, _ in targets]
    guesses = fill_mask(masked_texts, top_k=top_k, batch_size=32) if masked_texts else []
    if len(masked_texts) == 1:  # the pipeline drops the outer list for a single input
        guesses = [guesses]
    swaps = []  # (piece index, (new word, probability))
    for (i, tag), word_guesses in zip(targets, guesses):
        word = pieces[i]
        window_start = max(0, i - pos_window)
        window, pos_in_window = pieces[window_start:i + pos_window], i - window_start
        good = []
        for guess in word_guesses:
            new_word, prob = guess["token_str"].strip(), guess["score"]  # ModernBERT tokens start with a space
            if new_word == word or not new_word.isalpha():
                continue
            window[pos_in_window] = new_word
            same_pos = tagger.tag(window)[pos_in_window][1] == tag
            if same_pos and cosine(word_vectors.get_word_vector(word), word_vectors.get_word_vector(new_word)) >= min_cos and prob >= min_prob:
                good.append((new_word, prob))
        if good:
            swaps.append((i, max(good, key=lambda g: g[1])))
    chosen = sorted(swaps, key=lambda s: s[1][1], reverse=True)[:min(int(len(word_spans) * rate), len(swaps))]
    for i, (new_word, _) in chosen:
        pieces[i] = new_word
    return "".join(pieces), len(chosen)


ATTACKS = {  # name -> (function, rate as in RAID's get_attack); "none" = the same texts unattacked, for comparison
    "none": (lambda text, rate: (text, 0), 0),
    "alternative_spelling": (alternative_spelling, 1.0), "article_deletion": (article_deletion, 0.5),
    "insert_paragraphs": (insert_paragraphs, 0.5), "whitespace": (whitespace, 0.2), "glue": (glue, 0.05),
    "number": (number, 0.5), "misspelling": (misspelling, 0.2), "synonym": (synonym, 0.5),
}


def main(split, attack_names, n_texts=None):
    """Attack the machine texts of load_otb(split) (all, or a random sample of n_texts, seed 0) with seed 42 per attack
    like RAID's pipeline; write data/attacks/<split>_<attack>.jsonl with text_id, url, model, text, n_edits."""
    texts = [item for item in load_otb(split) if item[2] == 1]  # item = (url, text, label, model), label 1 = machine
    texts = random.Random(0).sample(texts, n_texts) if n_texts else texts
    ATTACK_DIR.mkdir(parents=True, exist_ok=True)
    for name in attack_names:
        attack, rate = ATTACKS[name]
        random.seed(42)
        with open(ATTACK_DIR / f"{split}_{name}.jsonl", "w") as out:
            for k, (url, text, _, model) in enumerate(texts):
                new_text, n_edits = attack(text, rate)
                out.write(json.dumps(dict(text_id=f"{name}_{k}", url=url, model=model, text=new_text, n_edits=n_edits)) + "\n")
                if k % 1000 == 0:
                    print(name, k, len(texts), flush=True)
        print("wrote", ATTACK_DIR / f"{split}_{name}.jsonl", flush=True)


if __name__ == "__main__":  # raid.py [split=test] [n_texts=40 | all] [attack ... = all]
    args = sys.argv[1:] + ["test", "40"][len(sys.argv) - 1:]  # fill in the missing defaults
    main(args[0], args[2:] or list(ATTACKS), None if args[1] == "all" else int(args[1]))
