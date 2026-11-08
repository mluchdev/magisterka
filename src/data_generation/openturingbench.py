"""OpenTuringBench (MLNTeam-Unical/OpenTuringBench), the default dataset: news articles, 7 open LLMs vs human.

The HF repo contains only machine texts; human articles are scraped with `fetch_otb_human.py` into
data/openturingbench/human_<split>.jsonl. A split without that file is loaded as machine-only.
"""

import json
import re
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

DATASET_ID = "MLNTeam-Unical/OpenTuringBench"
HUMAN_DIR = Path(__file__).parent.parent.parent / "data" / "openturingbench"
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


class OTBDataset(Dataset):
    """Items are (url, text, label, model): label 0 = human (model "human"), 1 = machine; deduplicated by text.
    With clean=True the machine-only header label lines (see strip_labels) are cut first.

    `url` is the article id shared by a human text and its 7 machine versions, so it is NOT unique per item.
    """

    def __init__(self, split: str = "test", config: str = "in_domain", require_human: bool = True, clean: bool = True):
        rows = list(load_dataset(DATASET_ID, config)[split].select_columns(["url", "content", "model"]))
        human_file = HUMAN_DIR / f"human_{split}.jsonl"
        if human_file.exists():
            rows += [json.loads(line) for line in human_file.read_text().splitlines() if line]
        elif require_human:
            raise FileNotFoundError(f"{human_file} missing: run `uv run python src/data_generation/fetch_otb_human.py {split}`")
        self.items, seen = [], set()
        for r in rows:
            text = strip_labels(r["content"]) if clean else r["content"]
            if text.strip() and text not in seen:
                seen.add(text)
                self.items.append((r["url"], text, int(r["model"] != "human"), r["model"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def otb_loader(split: str = "test", config: str = "in_domain", batch_size: int = 16, **kwargs) -> DataLoader:
    """Batches of (urls, texts, labels, models); labels is a tensor, the rest are lists/tuples of str."""
    return DataLoader(OTBDataset(split, config, **kwargs), batch_size=batch_size)


if __name__ == "__main__":
    ds = OTBDataset(require_human=False)
    print(len(ds), "items; first:", ds[0][0], ds[0][2], ds[0][3], repr(ds[0][1][:100]))
