import json
import re
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from common import DATA

DATASET_ID = "MLNTeam-Unical/OpenTuringBench"
HUMAN_DIR = DATA / "openturingbench"
HEADER_LINES = 8
# header lines like "Date: June 3, 2015" / "Category: LATINO VOICES": ~45% of machine texts, ~0% of human ones
LABEL = re.compile(r"^\s*(date|category|news category|subtitle|subheadline|headline|published|published on|publication|publication date)\s*:", re.I)


def strip_labels(text: str) -> str:
    """Drop machine-only "Label: value" lines from the first HEADER_LINES lines (and the blank lines they leave behind)."""
    lines = text.split("\n")
    out = []
    for line in [l for l in lines[:HEADER_LINES] if not LABEL.match(l)]:
        if not (out and not out[-1].strip() and not line.strip()):  # no double blank lines
            out.append(line)
    return "\n".join(out + lines[HEADER_LINES:])


# trailing boilerplate paragraphs (matched at paragraph start, case-insensitive): human = HuffPost promos/credits/
# corrections, machine = notes/disclaimers/placeholders/end markers. Mapped with find_artifacts.py on test.
FOOTER = re.compile(r"^[(\[]?\s*(note\b|editor[’']?s note|disclaimer|correction|update\b|end of (the )?article|your name|"
                    r"keep in touch|earlier on huff|want more\?|want to read|check (it )?out|like us on|related on huffpost|"
                    r"also on huffpost|do you have|have something to|for a constant|this (post|story|article) (was|has|originally)|"
                    r"(additional )?reporting by|writing by|editing by|follow .{0,60} on (twitter|facebook|instagram)|"
                    r"watch the (video|full)|click (here|through)|sign up for|to read more|you can read|a version of|for more on\b|about the author|need help\?|head over to|"
                    r"[-*#_=~ ]+$)", re.I)
CREDIT = re.compile(r"\s*\((additional )?(reporting|writing|editing)\b[^()]*\)\s*$", re.I)  # "... (Reporting by X; Editing by Y)"
MAX_FOOTER = 4


def normalize_ws(text: str) -> str:
    """Same whitespace as the human scraper (fetch_otb_human.clean_text): runs of whitespace inside a line -> one space,
    no leading/trailing spaces, empty lines dropped, paragraphs joined by exactly one blank line. Typography is untouched."""
    lines = (re.sub(r"\s+", " ", l).strip() for l in text.split("\n"))
    return "\n\n".join(l for l in lines if l)


def strip_footers(text: str) -> str:
    """Drop up to MAX_FOOTER trailing boilerplate paragraphs (FOOTER) and a trailing parenthesised credit (CREDIT)."""
    paras = text.split("\n\n")
    for _ in range(MAX_FOOTER):
        if len(paras) > 3 and FOOTER.match(paras[-1]):  # never eat the title/date header + first paragraph
            paras.pop()
    paras[-1] = CREDIT.sub("", paras[-1])
    return "\n\n".join(paras)


def clean_text(text: str) -> str:
    return strip_footers(normalize_ws(strip_labels(text)))


def load_otb(split="test", require_human=True):
    """OpenTuringBench in_domain: news, 7 open LLMs (HF repo) vs human (scraped by fetch_otb_human.py). List of
    (url, text, label, model): label 0 = human (model "human"), 1 = machine; cleaned by clean_text (both classes,
    typography kept), deduplicated by text. `url` is shared by a human text and its 7 machine versions."""
    rows = list(load_dataset(DATASET_ID, "in_domain", split=split).select_columns(["url", "content", "model"]))
    human_file = HUMAN_DIR / f"human_{split}.jsonl"
    if human_file.exists():
        rows += [json.loads(line) for line in human_file.read_text().splitlines() if line]
    elif require_human:
        raise FileNotFoundError(f"{human_file} missing: run `uv run python src/data_generation/fetch_otb_human.py {split}`")
    items = {}
    for r in rows:
        text = clean_text(r["content"])
        if text.strip():
            items.setdefault(text, (r["url"], text, int(r["model"] != "human"), r["model"]))
    return list(items.values())


if __name__ == "__main__":
    items = load_otb(require_human=False)
    print(len(items), "items; first:", items[0][0], items[0][2], items[0][3], repr(items[0][1][:100]))
