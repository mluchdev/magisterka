import csv
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from common import DATA
from data_generation.observer import VARIANTS, Observer
from data_generation.openturingbench import load_otb


def words(s):
    return re.findall(r"[\w'’-]+", s)


def ngrams4(s):
    w = [x.lower() for x in words(s)]
    return {tuple(w[i: i + 4]) for i in range(len(w) - 3)}


def copies(summary, article):
    """>= 30% of the summary's word 4-grams occur in the article."""
    p, a = ngrams4(summary), ngrams4(article)
    return bool(p) and len(p & a) / len(p) >= 0.3


def invented(summary, article):
    """Numbers / capitalised words (not at a sentence start) of the summary that do not occur in the article."""
    strip = lambda w: re.sub(r"['’]s$", "", w.lower())  # "Trump's" counts as "Trump"
    known = {strip(x) for x in words(article)}
    starts = {m.start() for m in re.finditer(r"(?:^|[.!?]\s+|\n)", summary)}
    return [m.group() for m in re.finditer(r"[\w'’-]+", summary)
            if (m.group()[0].isupper() or m.group()[0].isdigit()) and not any(abs(m.start() - s) <= 2 for s in starts)
            and strip(m.group()) not in known]


def main(n=60):
    sample = random.Random(1).sample(load_otb("test"), n)
    obs = Observer()
    texts, ids, _ = zip(*(obs.cut(t) for _, t, *_ in sample))
    is_name = [np.array([bool(re.match(r"\s*[A-Z0-9]", obs.tok.decode([t]))) for t in i]) for i in ids]
    rows = []
    print(f"{'variant':10}{'empty':>7}{'copy':>7}{'invent':>8}{'words':>7}{'p(all)':>8}{'p(names)':>10}")
    for variant in VARIANTS:
        obs.variant = variant
        summ = obs.summarize(texts)
        arrs = obs.score(summ, ids)
        p_all = np.mean([a[:, -1].mean() for a in arrs])
        p_names = np.mean([a[m, -1].mean() for a, m in zip(arrs, is_name)])
        st = [(s, t, copies(s, t), invented(s, t)) for s, t in zip(summ, texts)]
        m = np.mean([(not s, c, bool(inv), len(s.split())) for s, _, c, inv in st], axis=0)
        print(f"{variant:10}{m[0]:7.1%}{m[1]:7.1%}{m[2]:8.1%}{m[3]:7.0f}{p_all:8.3f}{p_names:10.3f}")
        rows += [[variant, t[:300], s, int(c), "|".join(inv)] for s, t, c, inv in st]
    out = DATA / "reverse_prompt_validation.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows([["variant", "article_start", "summary", "copy", "invented"], *rows])
    print("saved", out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
