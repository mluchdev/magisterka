import difflib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from common import DATA, OBSERVER_MODEL
from data.observer_data import OBS_DIR, read_index
from data_generation.observer_stats import text_stats

OUT = DATA / "figures_attack"
STATS = ["perplexity", "top20_hit", "gltr_>20", "entropy", "curvature"]


def load(name):
    """Per-text stats + token arrays of an observer split; texts in attack order (text_id <attack>_<k>)."""
    idx = read_index(name, text=True)
    arr = np.load(OBS_DIR / f"{name}.npy", mmap_mode="r")
    idx["k"] = idx.text_id.str.rsplit("_", n=1).str[1].astype(int)
    idx = idx.sort_values("k").reset_index(drop=True)
    toks = [np.asarray(arr[s:e]) for s, e in zip(idx.row_start, idx.row_end)]
    return pd.concat([idx[["k", "text", "group"]], pd.DataFrame([text_stats(a)[0] for a in toks])], axis=1), toks


def show(tok, base, att, base_p, att_p, w=6):
    """First edited region: tokens of the original and the attacked text with Nemotron's p(actual token)."""
    a, b = tok(base, add_special_tokens=False).input_ids[:len(base_p)], tok(att, add_special_tokens=False).input_ids[:len(att_p)]
    op = next((o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != "equal"), None)
    if op is None:
        return "   (unchanged)"
    fmt = lambda ids, p, s, e: " ".join(f"{tok.decode([t])!r}:{q:.2f}" for t, q in zip(ids[max(0, s - w):e + w], p[max(0, s - w):e + w]))
    return f"   before: {fmt(a, base_p, op[1], op[2])}\n   after:  {fmt(b, att_p, op[3], op[4])}"


def main(split, names, example=0):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(OBSERVER_MODEL)
    OUT.mkdir(parents=True, exist_ok=True)
    data = {n: load(f"{split}_atk_{n}") for n in ["none", *names]}
    hum = read_index(split).query("label == 0").sample(400, random_state=0)  # human reference (observer run on <split>)
    arr = np.load(OBS_DIR / f"{split}.npy", mmap_mode="r")
    data["human (ref)"] = (pd.DataFrame([text_stats(np.asarray(arr[s:e]))[0] for s, e in zip(hum.row_start, hum.row_end)]), None)

    base, base_toks = data["none"]
    rows = []
    for n, (df, toks) in data.items():
        r = {"attack": n, "n_texts": len(df), **{s: df[s].mean() for s in STATS}}
        if toks is not None and n != "none":
            r |= {f"d_{s}": (df[s] - base[s]).mean() for s in ("perplexity", "top20_hit")}  # paired change vs unattacked
            print(f"\n=== {n}: example text {example}, Nemotron p(actual token) around the first edit")
            print(show(tok, base.text[example], df.text[example], base_toks[example][:, 20], toks[example][:, 20]))
        rows.append(r)
    table = pd.DataFrame(rows).set_index("attack")
    print("\nmean per-text stats (machine texts; human (ref) = human texts of the original observer run):\n", table.round(3).to_string())
    table.to_csv(OUT / f"attack_{split}_stats.csv")

    names_all = list(data)
    fig, axs = plt.subplots(1, len(STATS), figsize=(5 * len(STATS), 5))
    for ax, s in zip(axs, STATS):
        ax.boxplot([data[n][0][s] for n in names_all], showfliers=False)
        ax.set_xticks(range(1, len(names_all) + 1), names_all, rotation=60, ha="right")
        ax.set(title=s, yscale="log" if s == "perplexity" else "linear")
    fig.suptitle(f"Nemotron observer per-text stats, machine texts before/after attack ({split}, {len(base)} texts)")
    fig.tight_layout()
    fig.savefig(OUT / f"attack_{split}_stats.png", dpi=130)

    fig, ax = plt.subplots(figsize=(9, 5))
    for n, (df, toks) in data.items():
        if toks is not None:
            ax.hist(-np.log(np.concatenate([t[:, 20] for t in toks])), np.linspace(0, 20, 81), density=True, histtype="step", lw=1.5, label=n,
                    color="black" if n == "none" else None)
    ax.set(yscale="log", xlabel="surprisal of the actual token (nats)", ylabel="density", title="Token surprisal under Nemotron, before/after attack")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"attack_{split}_token_surprisal.png", dpi=130)
    print("figures in", OUT)


if __name__ == "__main__":  # compare.py [split=test] [attack ... = every scored attack of the split]
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    main(split, sys.argv[2:] or sorted({p.name[len(split) + 5:-10] for p in OBS_DIR.glob(f"{split}_atk_*_index.csv")} - {"none"}))
