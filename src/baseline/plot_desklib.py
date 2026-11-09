import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from common import DATA, ROOT, roc

NAME = "desklib_openturingbench_test"
TITLE = "desklib on OpenTuringBench test (human vs AI)"
HUMAN, AI = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a19", "#6b6a64", "#e4e3de"
OUT = ROOT / "graphs"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})


def save(fig, suffix):
    fig.tight_layout()
    fig.savefig(OUT / f"{NAME}_{suffix}.png", dpi=150)


def plot_roc(y, s, m):
    fpr, tpr = roc(y, s)
    fig, (lin, log) = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax in (lin, log):
        ax.plot(fpr, tpr, color=HUMAN, lw=2, label=f"desklib (AUROC = {m['auroc']:.3f})")
        ax.set(xlabel="False positive rate (human texts flagged as AI)", ylabel="True positive rate (AI texts detected)", ylim=(0, 1.01))
    lin.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="random")
    lin.set(xlim=(0, 1), title="Full range")
    lin.legend(loc="lower right", frameon=False)
    log.set(xscale="log", xlim=(1e-3, 1), title="Low-FPR region (log scale)")
    for f, key in ((0.01, "tpr_at_1pct_fpr"), (0.001, "tpr_at_0.1pct_fpr")):
        log.axvline(f, color=MUTED, lw=1, ls=":")
        log.annotate(f"TPR = {m[key]:.3f}\n@ {f:.1%} FPR", (f, m[key]), xytext=(6, 30), textcoords="offset points", color=INK, fontsize=9)
    fig.suptitle(f"ROC curve: {TITLE}")
    save(fig, "roc")


def plot_confusion(m):
    cm = np.array(m["confusion_matrix"])
    rows = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({rows[i, j]:.1%})", ha="center", va="center", color="white" if rows[i, j] > 0.5 else INK, fontsize=12)
    ax.set_xticks([0, 1], ["human", "AI"])
    ax.set_yticks([0, 1], ["human", "AI"])
    ax.set(xlabel="Predicted", ylabel="True")
    ax.set_title(f"Confusion matrix (threshold {m['threshold']}, F1 = {m['f1']:.3f})\n{TITLE}", fontsize=11)
    save(fig, "confusion_matrix")


def plot_scores(y, s, m):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for lab, color, name, n in ((0, HUMAN, "human", m["n_human"]), (1, AI, "AI", m["n_ai"])):
        ax.hist(s[y == lab], bins=np.linspace(0, 1, 41), color=color, alpha=0.7, edgecolor="#fcfcfb", label=f"{name} (n = {n})")
    ax.axvline(m["threshold"], color=INK, lw=1, ls="--", label=f"threshold {m['threshold']}")
    ax.set(xlabel="desklib score, P(AI-generated)", ylabel="Number of texts")
    ax.set_title(f"Score distribution: {TITLE}", fontsize=11)
    ax.legend(frameon=False)
    save(fig, "score_distribution")


if __name__ == "__main__":
    m = json.loads((DATA / f"{NAME}_metrics.json").read_text())
    df = pd.read_csv(DATA / f"{NAME}_scores.csv")
    y, s = df.label.to_numpy(), df.score.to_numpy()
    OUT.mkdir(exist_ok=True)
    plot_roc(y, s, m); plot_confusion(m); plot_scores(y, s, m)
    print(*sorted(OUT.glob(f"{NAME}_*.png")), sep="\n")
