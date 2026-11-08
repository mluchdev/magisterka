"""Compare ways of making Nemotron summarise an article (observer.PROMPTS + "none" = one generic prompt for all), on the
same N test articles.

Per variant: empty replies, summaries that copy the article (>= 30% of the summary's word 4-grams occur in the article),
summaries with an invented entity (a number or capitalised word that is not in the article), mean words, and how well
the summary prepares the model: mean p(actual token) over the article, and over its names/numbers only (tokens that
start with a capital letter or digit). All summaries are saved to data/reverse_prompt_validation.csv for reading.
Run: .venv-vllm/bin/python src/data_generation/validate_reverse_prompt.py [n_articles]
"""

import csv
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from data_generation.observer import PROMPTS, ask, cut, load_llm, score
from data_generation.openturingbench import OTBDataset


def words(s):
    return re.findall(r"[\w'’-]+", s)


def copies(summary, article):
    ng = lambda w: {tuple(w[i : i + 4]) for i in range(len(w) - 3)}
    p, a = ng([x.lower() for x in words(summary)]), ng([x.lower() for x in words(article)])
    return len(p) > 0 and len(p & a) / len(p) >= 0.3


def invented(summary, article):
    """Numbers / capitalised words (not at a sentence start) of the summary that do not occur in the article."""
    strip = lambda w: re.sub(r"['’]s$", "", w.lower())  # "Trump's" counts as "Trump"
    known = {strip(x) for x in words(article)}
    starts = {m.start() for m in re.finditer(r"(?:^|[.!?]\s+|\n)", summary)}
    out = []
    for m in re.finditer(r"[\w'’-]+", summary):
        w = m.group()
        if (w[0].isupper() or w[0].isdigit()) and not any(abs(m.start() - s) <= 2 for s in starts) and strip(w) not in known:
            out.append(w)
    return out


def main(n=60):
    items = OTBDataset("test").items
    sample = random.Random(1).sample(items, n)
    llm, tok = load_llm()
    cuts = [cut(tok, t) for _, t, *_ in sample]
    texts, ids = [c[0] for c in cuts], [c[1] for c in cuts]
    is_name = [np.array([bool(re.match(r"\s*[A-Z0-9]", tok.decode([t]))) for t in i]) for i in ids]
    rows = []
    print(f"{'variant':10}{'empty':>7}{'copy':>7}{'invent':>8}{'words':>7}{'p(all)':>8}{'p(names)':>10}")
    for variant in [*PROMPTS, "none"]:
        summ = ask(llm, texts, variant)
        arrs = score(llm, tok, summ, ids, variant)
        p_all = np.mean([a[:, -1].mean() for a in arrs])
        p_names = np.mean([a[m, -1].mean() for a, m in zip(arrs, is_name)])
        st = [(not s, copies(s, t), bool(invented(s, t)), len(s.split())) for s, t in zip(summ, texts)]
        m = np.mean(st, axis=0)
        print(f"{variant:10}{m[0]:7.1%}{m[1]:7.1%}{m[2]:8.1%}{m[3]:7.0f}{p_all:8.3f}{p_names:10.3f}")
        rows += [[variant, t[:300], s, int(c), "|".join(invented(s, t))] for t, s, (_, c, _, _) in zip(texts, summ, st)]
    out = Path(__file__).parent.parent.parent / "data" / "reverse_prompt_validation.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows([["variant", "article_start", "summary", "copy", "invented"], *rows])
    print("saved", out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
