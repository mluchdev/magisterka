from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
OBSERVER_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"


def ranks(s):
    """1-based ranks, ties get their average rank."""
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2)[inv]


def auroc(y, s):
    """y 0/1, s scores (higher = machine). Mann-Whitney U, ties count 1/2."""
    n1, n0 = y.sum(), (1 - y).sum()
    return float((ranks(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tpr_at_fpr(y, s, fpr=0.01):
    """TPR with the threshold set so that at most `fpr` of human texts (y == 0) are flagged."""
    return float((s[y == 1] > np.quantile(s[y == 0], 1 - fpr, method="higher")).mean())


def roc(y, s):
    """FPR and TPR at every score threshold, from high to low."""
    y = y[np.argsort(-s)]
    return np.r_[0, np.cumsum(1 - y) / (1 - y).sum()], np.r_[0, np.cumsum(y) / y.sum()]
