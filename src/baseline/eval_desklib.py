import csv
import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from baseline.desklib import MODEL_ID, detect
from common import DATA, auroc, tpr_at_fpr
from data_generation.openturingbench import load_otb

SPLIT = "test"
THRESHOLD = 0.5
NAME = f"desklib_openturingbench_{SPLIT}"


def main():
    rows = []
    for urls, texts, labels, models in DataLoader(load_otb(SPLIT), 16):
        rows += zip(urls, models, labels.tolist(), detect(texts))
        print(f"{len(rows)} texts scored", end="\r")
    urls, models, y, s = map(np.array, zip(*rows))

    pred = (s >= THRESHOLD).astype(int)
    cm = np.zeros((2, 2), int)  # rows = true (human, ai), cols = predicted (human, ai)
    np.add.at(cm, (y, pred), 1)
    (tn, fp), (fn, tp) = cm.tolist()
    gens = sorted(set(models) - {"human"})
    one_vs_human = lambda m: (models == m) | (y == 0)
    results = {
        "model": MODEL_ID,
        "dataset": f"OpenTuringBench in_domain {SPLIT}, human (scraped) vs 7 LLMs, dedup by text",
        "n_human": tn + fp,
        "n_ai": tp + fn,
        "threshold": THRESHOLD,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "auroc": auroc(y, s),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, s, 0.01),
        "tpr_at_0.1pct_fpr": tpr_at_fpr(y, s, 0.001),
        "confusion_matrix": cm.tolist(),
        # per generator: TPR at the threshold, and at the threshold giving 1% FPR on human texts
        "per_model_tpr": {m: float((s[models == m] >= THRESHOLD).mean()) for m in gens},
        "per_model_tpr_at_1pct_fpr": {m: tpr_at_fpr(y[one_vs_human(m)], s[one_vs_human(m)], 0.01) for m in gens},
    }

    DATA.mkdir(exist_ok=True)
    (DATA / f"{NAME}_metrics.json").write_text(json.dumps(results, indent=2))
    with open(DATA / f"{NAME}_scores.csv", "w", newline="") as f:
        csv.writer(f).writerows([("url", "model", "label", "score"), *rows])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
