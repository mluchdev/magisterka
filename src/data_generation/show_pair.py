import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from data.observer_data import read_index

split = sys.argv[1] if len(sys.argv) > 1 else "test"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
idx = read_index(split, text=True)
urls = sorted(set(idx.url[idx.label == 0]) & set(idx.url[idx.label == 1]))
while True:
    g = idx[idx.url == random.choice(urls)]
    pair = [g[g.label == 0].iloc[0], g[g.label == 1].sample(1).iloc[0]]
    random.shuffle(pair)
    for name, r in zip("AB", pair):
        t = r.text if len(r.text) <= 2 * n else r.text[:n] + " ... " + r.text[-n:]
        print(f"\n===== {name} ({len(r.text)} chars) =====\n{t!r}")
    if input("\nEnter = reveal, q = quit: ").strip() == "q":
        break
    for name, r in zip("AB", pair):
        print(f"{name}: {'HUMAN' if r.label == 0 else 'MACHINE ' + r.group}")
    if input("Enter = next, q = quit: ").strip() == "q":
        break
