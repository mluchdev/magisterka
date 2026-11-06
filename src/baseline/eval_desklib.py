"""Evaluate desklib on OpenTuringBench (human vs 7 open LLMs); results go to data/."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from baseline.desklib import MODEL_ID, desklib_detect_batch
from data.openturingbench import otb_loader

SPLIT = "test"
THRESHOLD = 0.5
OUT = Path(__file__).parent.parent.parent / "data"


def auroc(y, s):
    """Mann-Whitney U formulation, ties counted as 1/2."""
    pos, neg = s[y == 1], s[y == 0]
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return greater / (len(pos) * len(neg))


def tpr_at_fpr(y, s, fpr):
    """TPR with the threshold set so that at most `fpr` of human texts are flagged."""
    thr = np.quantile(s[y == 0], 1 - fpr, method="higher")
    return float((s[y == 1] > thr).mean())


def main():
    ids, models, labels, scores = [], [], [], []
    for batch_ids, texts, batch_labels, batch_models in otb_loader(SPLIT):
        ids += batch_ids
        models += batch_models
        labels += batch_labels.tolist()
        scores += desklib_detect_batch(list(texts))
        print(f"{len(ids)} texts scored", end="\r")

    y, s, models = np.array(labels), np.array(scores), np.array(models)
    pred = (s >= THRESHOLD).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    results = {
        "model": MODEL_ID,
        "dataset": f"OpenTuringBench in_domain {SPLIT}, human (scraped) vs 7 LLMs, dedup by text",
        "n_human": tn + fp,
        "n_ai": tp + fn,
        "threshold": THRESHOLD,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "auroc": float(auroc(y, s)),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, s, 0.01),
        "tpr_at_0.1pct_fpr": tpr_at_fpr(y, s, 0.001),
        # rows = true (human, ai), cols = predicted (human, ai)
        "confusion_matrix": [[tn, fp], [fn, tp]],
        # per generator: TPR at the threshold, and at the threshold giving 1% FPR on human texts
        "per_model_tpr": {m: float((s[models == m] >= THRESHOLD).mean()) for m in sorted(set(models) - {"human"})},
        "per_model_tpr_at_1pct_fpr": {
            m: float((s[models == m] > np.quantile(s[y == 0], 0.99, method="higher")).mean())
            for m in sorted(set(models) - {"human"})
        },
    }

    OUT.mkdir(exist_ok=True)
    name = f"desklib_openturingbench_{SPLIT}"
    (OUT / f"{name}_metrics.json").write_text(json.dumps(results, indent=2))
    with open(OUT / f"{name}_scores.csv", "w", newline="") as f:
        csv.writer(f).writerows([("url", "model", "label", "score"), *zip(ids, models, labels, scores)])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
