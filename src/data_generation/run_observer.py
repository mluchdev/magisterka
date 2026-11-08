"""Run observer.py until the split is finished; after a crash (typically CUDA out of memory) restart it with safer settings.

The observer is resumable, so every restart continues after the last saved chunk. Settings ladder (OBS_* env vars of
observer.py): each crash without progress moves one step down (the last step is repeated); a crash after progress restarts at the top. Gives up after MAX_CRASHES crashes in a row
without any progress. Ctrl+C (or `kill -INT <pid of this script>`) is forwarded once to the observer, which saves and
exits; this script then stops too.
Run: .venv-vllm/bin/python src/data_generation/run_observer.py [split] [n_texts]
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
GPU_UTIL = 0.74  # fixed: it is a floor for the KV cache (0.70 leaves too little for a 6144-token sequence), not a safety margin
LADDER = [  # BATCHED_TOKENS, MAX_SEQS, CHUNK; the first step is tried twice (a crash may be a one-off fault)
    (256, 4, 8),
    (256, 4, 8),
    (128, 2, 4),
    (64, 1, 2),
]
MAX_CRASHES = 6


def main(args):
    split = args[0] if args else "test"
    npy = ROOT / "data" / "observer" / f"{split}.npy"
    child, stop, level, crashes = None, [], 0, 0

    def on_sigint(*_):
        stop.append(1)
        if child and child.poll() is None:
            child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, on_sigint)
    while True:
        batched, seqs, chunk = LADDER[min(level, len(LADDER) - 1)]
        util = GPU_UTIL
        env = {**os.environ, "OBS_GPU_UTIL": str(util), "OBS_BATCHED_TOKENS": str(batched), "OBS_MAX_SEQS": str(seqs), "OBS_CHUNK": str(chunk)}
        size_before = npy.stat().st_size if npy.exists() else 0
        print(f"[run_observer] start, level {level}: util={util} batched_tokens={batched} seqs={seqs} chunk={chunk}", flush=True)
        # own session: a terminal Ctrl+C reaches only this script, which forwards it once (a second one could cut a save short)
        child = subprocess.Popen([sys.executable, str(Path(__file__).parent / "observer.py"), *args], env=env, start_new_session=True)
        code = child.wait()
        subprocess.run(["pkill", "-f", "VLLM::EngineCore"])  # a crashed engine can keep holding GPU memory
        time.sleep(10)
        if stop:
            print("[run_observer] interrupted, stopped", flush=True)
            return 130
        if code == 0:
            print("[run_observer] finished", flush=True)
            return 0
        progress = (npy.stat().st_size if npy.exists() else 0) > size_before
        crashes = 0 if progress else crashes + 1
        level = 0 if progress else level + 1
        print(f"[run_observer] crashed (exit {code}), progress since start: {progress}, crashes without progress: {crashes}", flush=True)
        if crashes >= MAX_CRASHES:
            print("[run_observer] giving up", flush=True)
            return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
