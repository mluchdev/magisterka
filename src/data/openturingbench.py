"""OpenTuringBench (MLNTeam-Unical/OpenTuringBench), the default dataset: news articles, 7 open LLMs vs human.

The HF repo contains only machine texts; human articles are scraped with `fetch_otb_human.py` into
data/openturingbench/human_<split>.jsonl. A split without that file is loaded as machine-only.
"""

import json
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

DATASET_ID = "MLNTeam-Unical/OpenTuringBench"
HUMAN_DIR = Path(__file__).parent.parent.parent / "data" / "openturingbench"


class OTBDataset(Dataset):
    """Items are (url, text, label, model): label 0 = human (model "human"), 1 = machine; deduplicated by text.

    `url` is the article id shared by a human text and its 7 machine versions, so it is NOT unique per item.
    """

    def __init__(self, split: str = "test", config: str = "in_domain", require_human: bool = True):
        rows = list(load_dataset(DATASET_ID, config)[split].select_columns(["url", "content", "model"]))
        human_file = HUMAN_DIR / f"human_{split}.jsonl"
        if human_file.exists():
            rows += [json.loads(line) for line in human_file.read_text().splitlines() if line]
        elif require_human:
            raise FileNotFoundError(f"{human_file} missing: run `uv run python src/data/fetch_otb_human.py {split}`")
        self.items, seen = [], set()
        for r in rows:
            if r["content"].strip() and r["content"] not in seen:
                seen.add(r["content"])
                self.items.append((r["url"], r["content"], int(r["model"] != "human"), r["model"]))

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
