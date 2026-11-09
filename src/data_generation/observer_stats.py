import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from common import DATA
from data.observer_data import OBS_DIR, read_index

OUT = DATA / "figures"
BUCKETS = ["top-1", "2-5", "6-10", "11-20", ">20"]
QS = (10, 25, 50, 75, 90)
GAP_BINS = np.linspace(0, 1, 51)


def text_stats(a):
    """a: (n, 21) = top-20 p (descending, not renormalised) + p(actual). Per-text scalars, GLTR bucket shares and the
    token-level gap histogram. Entropy and curvature use only the top-20."""
    top, p = a[:, :20].astype(np.float64), a[:, 20].astype(np.float64)
    s = -np.log(p)  # surprisal, nats
    m, sd = s.mean(), s.std()
    logt = np.log(np.maximum(top, 1e-30))
    ent = -(top * logt).sum(1)  # truncated entropy
    var = (top * logt ** 2).sum(1) - ent ** 2  # variance of log p under top-20
    hit = p >= top[:, 19]
    rank = 1 + (top > p[:, None]).sum(1)  # valid only where hit
    gap = top[:, 0] - p
    b = np.where(hit, np.digitize(rank, [2, 6, 11]), 4)  # 0: rank 1, 1: 2-5, 2: 6-10, 3: 11-20, 4: outside top-20
    out = dict(n_tokens=len(a), perplexity=np.exp(m), surp_mean=m, surp_std=sd,
               surp_skew=((s - m) ** 3).mean() / sd ** 3 if sd > 0 else 0.0,
               **{f"surp_q{k}": v for k, v in zip(QS, np.quantile(s, np.array(QS) / 100))},
               top20_hit=hit.mean(), entropy=ent.mean(), top1_conf=top[:, 0].mean(), gap=gap.mean(),
               curvature=(np.log(p) + ent).sum() / np.sqrt(var.sum()),
               **{"gltr_" + k: v for k, v in zip(BUCKETS, np.bincount(b, minlength=5) / len(a))})
    return out, np.histogram(gap, GAP_BINS)[0]


def main(split):
    OUT.mkdir(parents=True, exist_ok=True)
    idx, arr = read_index(split), np.load(OBS_DIR / f"{split}.npy", mmap_mode="r")
    groups = sorted(set(idx.group))
    rows, gaps = [], {g: np.zeros(len(GAP_BINS) - 1) for g in groups}
    for r in idx.itertuples():
        st, gh = text_stats(np.asarray(arr[r.row_start:r.row_end]))
        rows.append(st)
        gaps[r.group] += gh
    df = pd.concat([idx[["text_id", "group", "label"]], pd.DataFrame(rows)], axis=1)
    df.to_csv(OUT / f"observer_{split}_text_stats.csv", index=False)
    num = df.drop(columns=["text_id", "label"]).groupby("group").agg(["mean", "median"])
    num.columns = ["_".join(c) for c in num.columns]
    num["n_texts"] = df.groupby("group").size()
    num.to_csv(OUT / f"observer_{split}_group_means.csv")
    colors = {g: ("black" if g == "human" else plt.cm.tab10(i)) for i, g in enumerate(groups)}

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(OUT / f"observer_{split}_{name}.png", dpi=130)
        plt.close(fig)

    def dist(ax, col, title, log=False):
        lo, hi = np.percentile(df[col], [0.5, 99.5])
        edges = np.geomspace(lo, hi, 60) if log else np.linspace(lo, hi, 60)
        for g in groups:
            ax.hist(df.loc[df.group == g, col].clip(lo, hi), edges, density=True, histtype="step", lw=1.5, color=colors[g], label=g)
        ax.set(xlabel=col, ylabel="density", title=title, xscale="log" if log else "linear")

    def one(col, name, title, log=False):
        fig, ax = plt.subplots(figsize=(8, 5))
        dist(ax, col, title, log)
        ax.legend(fontsize=8)
        save(fig, name)

    one("perplexity", "perplexity", "Perplexity per text (exp of mean surprisal)", log=True)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    dist(axs[0], "surp_std", "Surprisal standard deviation per text")
    for g in groups:
        axs[1].plot(QS, [df.loc[df.group == g, f"surp_q{k}"].mean() for k in QS], "o-", color=colors[g], label=g)
    axs[1].set(xlabel="quantile of token surprisal (%)", ylabel="mean over texts (nats)", title="Surprisal quantiles")
    dist(axs[2], "surp_skew", "Surprisal skewness per text")
    axs[0].legend(fontsize=8)
    save(fig, "surprisal_shape")
    one("top20_hit", "top20_hit_rate", "Top-20 hit rate per text (share of tokens inside the top-20)")

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
    one("entropy", "entropy", "Mean predictive entropy per text (nats, top-20 only)")
    one("top1_conf", "top1_confidence", "Mean top-1 probability per text")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    dist(axs[0], "gap", "Mean gap per text: p(top-1) - p(actual)")
    for g in groups:
        axs[1].plot((GAP_BINS[1:] + GAP_BINS[:-1]) / 2, gaps[g] / gaps[g].sum() / np.diff(GAP_BINS), color=colors[g], label=g)
    axs[1].set(yscale="log", xlabel="p(top-1) - p(actual)", ylabel="density", title="Gap per token (all tokens pooled)")
    axs[0].legend(fontsize=8)
    save(fig, "gap_to_top")
    one("curvature", "fast_detectgpt_curvature",
        "Fast-DetectGPT analytic curvature per text\n(sampling = scoring model; distribution truncated to the top-20)")
    print(num.filter(regex="mean$").round(3).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
