from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

DATASET_ID = "OpAI-Bench1/OpAI-Bench"


class PureDataset(Dataset):
    """OpAI-Bench texts that are fully human (label 0) or fully AI (label 1), deduplicated by text."""

    def __init__(self, split: str = "test"):
        data = load_dataset(DATASET_ID, "default")[split]
        data = data.select_columns(["record_id", "text", "ai_char_ratio"])
        self.items, seen = [], set()
        for r in data:
            if r["ai_char_ratio"] in (0.0, 1.0) and r["text"] not in seen:
                seen.add(r["text"])
                self.items.append((r["record_id"], r["text"], int(r["ai_char_ratio"])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class VersionDataset(Dataset):
    """OpAI-Bench texts labelled by edit version (0 = human ... 8 = last AI edit), deduplicated by text."""

    def __init__(self, split: str = "train", per_class: int | None = None, seed: int = 0):
        data = load_dataset(DATASET_ID, "default")[split]
        data = data.select_columns(["record_id", "text", "version_index"]).shuffle(seed=seed)
        self.items, seen, counts = [], set(), [0] * 9
        for r in data:
            v = r["version_index"]
            if (per_class is None or counts[v] < per_class) and r["text"] not in seen:
                seen.add(r["text"])
                counts[v] += 1
                self.items.append((r["record_id"], r["text"], v))
            if per_class is not None and min(counts) == per_class:
                break

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def version_loader(split: str = "train", per_class: int | None = None, batch_size: int = 8, shuffle: bool = False) -> DataLoader:
    """Batches of (record_ids, texts, versions)."""
    return DataLoader(VersionDataset(split, per_class), batch_size=batch_size, shuffle=shuffle)


def pure_loader(split: str = "test", batch_size: int = 16) -> DataLoader:
    """Batches of (record_ids, texts, labels)."""
    return DataLoader(PureDataset(split), batch_size=batch_size)


if __name__ == "__main__":
    train = load_dataset(DATASET_ID, "default")["train"]
    for key, value in train[0].items():
        print(f"{key}: {value}")
