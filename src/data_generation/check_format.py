"""Formatting check of OpenTuringBench test: do human and machine texts differ in surface format (not style)?

Prints the first 300 chars of a few texts per class, then per-class rates of simple format cues.
Run: uv run python src/data_generation/check_format.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from data_generation.openturingbench import OTBDataset

DATE = re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\s*$", re.M)
CUES = {
    "date line in first 3 lines": lambda t: bool(DATE.search("\n".join(t.split("\n")[:3]))),
    "first line < 100 chars (title-like)": lambda t: len(t.split("\n")[0]) < 100,
    "has blank-line paragraph breaks": lambda t: "\n\n" in t,
    "has curly quotes/apostrophes": lambda t: any(c in t for c in "‘’“”"),
    "has straight quotes/apostrophes": lambda t: "'" in t or '"' in t,
    "has non-ASCII chars": lambda t: not t.isascii(),
    "has markdown (* # _ `)": lambda t: bool(re.search(r"[*#`]|__", t)),
    "leading/trailing whitespace": lambda t: t != t.strip(),
    "bracket placeholder [Your ...]": lambda t: bool(re.search(r"\[(Your|Insert|Name|Source|Author)[^\]]{0,40}\]", t, re.I)),
}


def main():
    ds = OTBDataset("test")
    groups = defaultdict(list)
    for _, text, label, model in ds.items:
        groups["human" if label == 0 else "machine"].append(text)
        if label == 1:
            groups[model].append(text)

    for name in ("human", "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2-9b-it"):
        for t in groups[name][:2]:
            print(f"--- {name} ---\n{t[:300]!r}\n")

    names = ["human", "machine", *sorted(k for k in groups if k not in ("human", "machine"))]
    print(f"{'':38}" + "".join(f"{n.split('/')[-1][:14]:>15}" for n in names))
    print(f"{'n':38}" + "".join(f"{len(groups[n]):>15}" for n in names))
    print(f"{'median chars':38}" + "".join(f"{int(np.median([len(t) for t in groups[n]])):>15}" for n in names))
    for cue, f in CUES.items():
        print(f"{cue:38}" + "".join(f"{np.mean([f(t) for t in groups[n]]):>15.1%}" for n in names))


if __name__ == "__main__":
    main()
