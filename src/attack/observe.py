import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from common import DATA
from data_generation.observer import Observer, run


def main(split, names):
    """Nemotron observer on data/attacks/<split>_<attack>.jsonl -> data/observer/<split>_atk_<attack>.* (vLLM venv)."""
    obs = Observer()
    for name in names:
        rows = [json.loads(l) for l in (DATA / "attacks" / f"{split}_{name}.jsonl").read_text().splitlines()]
        run(f"{split}_atk_{name}", [(r["text_id"], r["url"], r["text"], 1, r["model"]) for r in rows], obs)


if __name__ == "__main__":  # .venv-vllm/bin/python src/attack/observe.py [split=test] [attack ... = every attack file of the split]
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    main(split, sys.argv[2:] or sorted(p.stem[len(split) + 1:] for p in (DATA / "attacks").glob(f"{split}_*.jsonl")))
