import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from common import DATA
from data.observer_data import read_index

OUT = DATA / "artifacts"
MIN_RATE, TOP, SHOW = 0.005, 100, 30
WORD = re.compile(r"\w+|[^\w\s]")
REGEX = {
    "leading whitespace": r"\A\s",
    "trailing whitespace": r"\s\Z",
    "double space": r"  ",
    "space before newline": r" \n",
    "3+ newlines": r"\n\n\n",
    "single newline between lines": r"[^\n]\n[^\n]",
    "glued sentence (a.B)": r"[a-z][.?!][A-Z][a-z]",
    "space before punctuation": r"\w [,.;:!?]",
    "bracket placeholder [..]": r"\[[^\]]{1,40}\]",
    "markdown bold/heading": r"\*\*|^#+ ",
    "bullet line": r"(?m)^\s*[-*•] ",
    "numbered line": r"(?m)^\s*\d+[.)] ",
    "URL": r"https?://|www\.",
    "@handle": r"@\w",
    "#hashtag": r"(?<!\w)#\w",
    "ALL-CAPS line": r"(?m)^[^a-z\n]*[A-Z]{4,}[^a-z\n]*$",
    "ellipsis ...": r"\.\.\.",
    "parenthesised credit (By/Reporting/Editing)": r"\((?:Reporting|Editing|Writing|Additional reporting|By) ",
    "non-ASCII char": r"[^\x00-\x7f]",
}


def char_name(c):
    return f"{c!r} U+{ord(c):04X} {unicodedata.name(c, 'CONTROL' if ord(c) < 32 else '?')}"


def edge_ngrams(text):
    lines = [l for l in text.split("\n") if l.strip()]
    out = set()
    for pos, part in (("head", lines[:3]), ("tail", lines[-3:])):
        w = WORD.findall(" ".join(part).lower())
        for k in (1, 2, 3):
            out.update(f"{pos}: {' '.join(w[j:j + k])}" for j in range(len(w) - k + 1))
    return out


def snippet(text, s, w=40):
    i = text.find(s) if s else -1
    return "" if i < 0 else repr(text[max(0, i - w):i + len(s) + w])


def rank(df, sets, show=str, find=lambda i: "", keep_all=False):
    """Document frequency of each item per group (df.group, one item set per text); log_ratio =
    log((h+1)/(nh+2)) - log((m+1)/(nm+2)), > 0 = human-leaning. show(item) -> printable name, find(item) -> substring
    for the example snippet. Keeps items in >= MIN_RATE of either class and the TOP each way, unless keep_all."""
    groups, n = sorted(set(df.group)), df.group.value_counts()
    cnt, first = {g: Counter() for g in groups}, {}
    for k, (s, g) in enumerate(zip(sets, df.group)):
        cnt[g].update(s)
        for i in s:
            first.setdefault(i, k)
    hum, mac = cnt.pop("human"), sum(cnt.values(), Counter())
    nh, nm = n["human"], len(df) - n["human"]
    items = [i for i in first if keep_all or max(hum[i] / nh, mac[i] / nm) >= MIN_RATE]
    r = pd.DataFrame({"item": [show(i) for i in items], "human": [hum[i] / nh for i in items], "machine": [mac[i] / nm for i in items],
                      "log_ratio": [np.log((hum[i] + 1) / (nh + 2)) - np.log((mac[i] + 1) / (nm + 2)) for i in items],
                      **{g: [c[i] / n[g] for i in items] for g, c in cnt.items()},
                      "example": [snippet(df.text.iat[first[i]], find(i)) for i in items]}).sort_values("log_ratio", ascending=False)
    return r if keep_all else pd.concat([r.head(TOP), r.tail(TOP)]).drop_duplicates("item")


def main(split):
    df = read_index(split, text=True)
    texts = df.text.tolist()
    scans = {
        "chars": rank(df, [set(t) for t in texts], char_name, lambda c: c),
        "edge_ngrams": rank(df, [edge_ngrams(t) for t in texts]),
        "regex": rank(df, [{k for k, p in REGEX.items() if re.search(p, t)} for t in texts], keep_all=True),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    cols = ["item", "human", "machine", "log_ratio"]
    for name, r in scans.items():
        r.to_csv(OUT / f"{split}_{name}.csv", index=False)
        print(f"\n===== {name}: human-leaning =====\n{r[cols].head(SHOW).round(3).to_string(index=False)}")
        if name != "regex":
            print(f"\n===== {name}: machine-leaning =====\n{r[cols].tail(SHOW).iloc[::-1].round(3).to_string(index=False)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
