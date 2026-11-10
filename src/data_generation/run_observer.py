import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from common import DATA

MAX_CRASHES = 6


def main(args):
    """Run observer.py until the split is done; after a crash restart it (it resumes). Gives up after MAX_CRASHES crashes
    in a row without the .npy growing. Ctrl+C is forwarded once to the observer, which saves and exits."""
    npy = DATA / "observer" / f"{args[0] if args else 'test'}.npy"
    size = lambda: npy.stat().st_size if npy.exists() else 0
    child, stop, crashes = None, [], 0

    def on_sigint(*_):
        stop.append(1)
        if child and child.poll() is None:
            child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, on_sigint)
    while True:
        before = size()
        print("[run_observer] start", flush=True)
        # own session: a terminal Ctrl+C reaches only this script, which forwards it once (a second one could cut a save short)
        child = subprocess.Popen([sys.executable, str(Path(__file__).parent / "observer.py"), *args], start_new_session=True)
        code = child.wait()
        subprocess.run(["pkill", "-f", "VLLM::EngineCore"])  # a crashed engine can keep holding GPU memory
        time.sleep(10)
        if stop or code == 0:
            print("[run_observer]", "interrupted" if stop else "finished", flush=True)
            return 130 if stop else 0
        crashes = 0 if size() > before else crashes + 1
        print(f"[run_observer] crashed (exit {code}), crashes without progress: {crashes}", flush=True)
        if crashes >= MAX_CRASHES:
            print("[run_observer] giving up", flush=True)
            return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
