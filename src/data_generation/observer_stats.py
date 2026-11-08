"""Per-text statistics of the observer output (data/observer/<split>.npy) per generator, with figures.

Run: uv run python src/data_generation/observer_stats.py [split]
Writes data/figures/observer_<split>_*.png, observer_<split>_text_stats.csv (one row per text) and observer_<split>_group_means.csv.
Row = 21 floats: p of the 20 most likely tokens (descending, NOT renormalised) + p of the actual token.
Entropy and curvature use only the top-20 (a truncated approximation of the full-vocabulary values).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "data" / "figures"
BUCKETS = ["top-1", "2-5", "6-10", "11-20", ">20"]
GAP_BINS = np.linspace(0, 1, 51)


def text_stats(a):
    """a: (n, 21) float. Returns dict of per-text scalars, GLTR bucket shares and the token-level gap histogram."""
    top, p = a[:, :20].astype(np.float64), a[:, 20].astype(np.float64)
    s = -np.log(p)  # surprisal, nats
    m, sd = s.mean(), s.std()
    plogp = top * np.log(np.maximum(top, 1e-30))
    ent = -plogp.sum(1)  # truncated entropy
    var = (top * np.log(np.maximum(top, 1e-30)) ** 2).sum(1) - plogp.sum(1) ** 2  # variance of log p under top-20
    hit = p >= top[:, 19]
    rank = 1 + (top > p[:, None]).sum(1)  # valid only where hit
    q = np.quantile(s, [0.1, 0.25, 0.5, 0.75, 0.9])
    gap = top[:, 0] - p
    out = dict(n_tokens=len(a), perplexity=np.exp(m), surp_mean=m, surp_std=sd,
               surp_skew=((s - m) ** 3).mean() / sd ** 3 if sd > 0 else 0.0,
               surp_q10=q[0], surp_q25=q[1], surp_q50=q[2], surp_q75=q[3], surp_q90=q[4],
               top20_hit=hit.mean(), entropy=ent.mean(), top1_conf=top[:, 0].mean(), gap=gap.mean(),
               curvature=(np.log(p) - plogp.sum(1)).sum() / np.sqrt(var.sum()))
    edges = [2, 6, 11]
    b = np.where(hit, np.digitize(rank, edges), 4)  # 0: rank 1, 1: 2-5, 2: 6-10, 3: 11-20, 4: outside top-20
    for name, share in zip(BUCKETS, np.bincount(b, minlength=5) / len(a)):
        out["gltr_" + name] = share
    return out, np.histogram(gap, GAP_BINS)[0]


def dist_plot(ax, df, col, groups, colors, log=False, bins=60):
    x = df[col].to_numpy()
    lo, hi = np.percentile(x, [0.5, 99.5])
    edges = np.geomspace(lo, hi, bins) if log else np.linspace(lo, hi, bins)
    for g in groups:
        ax.hist(df.loc[df.group == g, col].clip(lo, hi), edges, density=True, histtype="step", lw=1.5, color=colors[g], label=g)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(col)
    ax.set_ylabel("density")


def main(split):
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(ROOT / "data" / "observer" / f"{split}_index.csv", usecols=["text_id", "model", "label", "row_start", "row_end"])
    npy = np.load(ROOT / "data" / "observer" / f"{split}.npy", mmap_mode="r")
    idx["group"] = np.where(idx.label == 0, "human", idx.model.str.split("/").str[-1])
    rows, gaps = [], {g: np.zeros(len(GAP_BINS) - 1) for g in idx.group.unique()}
    for r in idx.itertuples():
        st, gh = text_stats(np.asarray(npy[r.row_start:r.row_end]))
        rows.append(st)
        gaps[r.group] += gh
    df = pd.concat([idx[["text_id", "group", "label"]], pd.DataFrame(rows)], axis=1)
    df.to_csv(OUT / f"observer_{split}_text_stats.csv", index=False)
    num = df.drop(columns=["text_id", "label"]).groupby("group").agg(["mean", "median"])
    num.columns = ["_".join(c) for c in num.columns]
    num["n_texts"] = df.groupby("group").size()
    num.to_csv(OUT / f"observer_{split}_group_means.csv")

    groups = ["human"] + sorted(g for g in df.group.unique() if g != "human")
    colors = {g: ("black" if g == "human" else plt.cm.tab10(i)) for i, g in enumerate(groups)}
    pre = OUT / f"observer_{split}_"

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(f"{pre}{name}.png", dpi=130)
        plt.close(fig)

    def one(col, name, log=False, title=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        dist_plot(ax, df, col, groups, colors, log)
        ax.set_title(title or col)
        ax.legend(fontsize=8)
        save(fig, name)

    one("perplexity", "perplexity", log=True, title="Perplexity per text (exp of mean surprisal)")

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    dist_plot(axs[0], df, "surp_std", groups, colors)
    axs[0].set_title("Surprisal standard deviation per text")
    for g in groups:
        sub = df[df.group == g]
        axs[1].plot([10, 25, 50, 75, 90], [sub[f"surp_q{k}"].mean() for k in (10, 25, 50, 75, 90)], "o-", color=colors[g], label=g)
    axs[1].set(xlabel="quantile of token surprisal (%)", ylabel="mean over texts (nats)", title="Surprisal quantiles")
    dist_plot(axs[2], df, "surp_skew", groups, colors)
    axs[2].set_title("Surprisal skewness per text")
    axs[0].legend(fontsize=8)
    save(fig, "surprisal_shape")

    one("top20_hit", "top20_hit_rate", title="Top-20 hit rate per text (share of tokens inside the top-20)")

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(groups))
    for i, b in enumerate(BUCKETS):
        v = np.array([df.loc[df.group == g, "gltr_" + b].mean() for g in groups])
        ax.bar(groups, v, bottom=bottom, label=b, color=plt.cm.viridis(i / 4))
        bottom += v
    ax.set(ylabel="share of tokens", title="GLTR-style rank buckets of the actual token (only ranks up to 20 are stored)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="rank", fontsize=8)
    save(fig, "gltr_buckets")

    one("entropy", "entropy", title="Mean predictive entropy per text (nats, top-20 only)")
    one("top1_conf", "top1_confidence", title="Mean top-1 probability per text")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    dist_plot(axs[0], df, "gap", groups, colors)
    axs[0].set_title("Mean gap per text: p(top-1) - p(actual)")
    mid = (GAP_BINS[1:] + GAP_BINS[:-1]) / 2
    for g in groups:
        axs[1].plot(mid, gaps[g] / gaps[g].sum() / np.diff(GAP_BINS), color=colors[g], label=g)
    axs[1].set(yscale="log", xlabel="p(top-1) - p(actual)", ylabel="density", title="Gap per token (all tokens pooled)")
    axs[0].legend(fontsize=8)
    save(fig, "gap_to_top")

    one("curvature", "fast_detectgpt_curvature",
        title="Fast-DetectGPT analytic curvature per text\n(sampling = scoring model; distribution truncated to the top-20)")
    print(num.filter(regex="mean$").round(3).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
