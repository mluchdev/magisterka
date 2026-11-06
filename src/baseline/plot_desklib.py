"""Plot desklib results from data/ (written by eval_desklib.py) as PNGs in graphs/."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.parent
NAME = "desklib_openturingbench_test"
TITLE = "desklib on OpenTuringBench test (human vs AI)"
HUMAN, AI = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a19", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})


def roc(y, s):
    """FPR and TPR at every distinct score threshold, from high to low."""
    order = np.argsort(-s)
    y = y[order]
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / (1 - y).sum()])
    return fpr, tpr


def plot_roc(y, s, m, out):
    fpr, tpr = roc(y, s)
    fig, (lin, log) = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax in (lin, log):
        ax.plot(fpr, tpr, color=HUMAN, lw=2, label=f"desklib (AUROC = {m['auroc']:.3f})")
        ax.set_xlabel("False positive rate (human texts flagged as AI)")
        ax.set_ylabel("True positive rate (AI texts detected)")
        ax.set_ylim(0, 1.01)
    lin.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="random")
    lin.set_xlim(0, 1)
    lin.set_title("Full range")
    lin.legend(loc="lower right", frameon=False)
    log.set_xscale("log")
    log.set_xlim(1e-3, 1)
    for f, key in ((0.01, "tpr_at_1pct_fpr"), (0.001, "tpr_at_0.1pct_fpr")):
        log.axvline(f, color=MUTED, lw=1, ls=":")
        log.annotate(f"TPR = {m[key]:.3f}\n@ {f:.1%} FPR", (f, m[key]), xytext=(6, 30),
                     textcoords="offset points", color=INK, fontsize=9)
    log.set_title("Low-FPR region (log scale)")
    fig.suptitle(f"ROC curve: {TITLE}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def plot_confusion(m, out):
    cm = np.array(m["confusion_matrix"])
    rows = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({rows[i, j]:.1%})", ha="center", va="center",
                    color="white" if rows[i, j] > 0.5 else INK, fontsize=12)
    ax.set_xticks([0, 1], ["human", "AI"])
    ax.set_yticks([0, 1], ["human", "AI"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (threshold {m['threshold']}, F1 = {m['f1']:.3f})\n{TITLE}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def plot_scores(y, s, m, out):
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(s[y == 0], bins=bins, color=HUMAN, alpha=0.7, edgecolor="#fcfcfb", label=f"human (n = {m['n_human']})")
    ax.hist(s[y == 1], bins=bins, color=AI, alpha=0.7, edgecolor="#fcfcfb", label=f"AI (n = {m['n_ai']})")
    ax.axvline(m["threshold"], color=INK, lw=1, ls="--", label=f"threshold {m['threshold']}")
    ax.set_xlabel("desklib score, P(AI-generated)")
    ax.set_ylabel("Number of texts")
    ax.set_title(f"Score distribution: {TITLE}", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    m = json.loads((ROOT / "data" / f"{NAME}_metrics.json").read_text())
    with open(ROOT / "data" / f"{NAME}_scores.csv") as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r["label"]) for r in rows])
    s = np.array([float(r["score"]) for r in rows])

    out = ROOT / "graphs"
    out.mkdir(exist_ok=True)
    plot_roc(y, s, m, out / f"{NAME}_roc.png")
    plot_confusion(m, out / f"{NAME}_confusion_matrix.png")
    plot_scores(y, s, m, out / f"{NAME}_score_distribution.png")
    print(*sorted(out.glob(f"{NAME}_*.png")), sep="\n")


if __name__ == "__main__":
    main()
