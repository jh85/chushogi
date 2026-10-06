#!/usr/bin/env python3
"""AZ self-play/training loop orchestrator.

One eval server (GPU, --watch on the current checkpoint) + W self-play
workers (C++ chushogi-az, pipe eval) + the trainer. Each iteration:

  1. workers play G games each against the current checkpoint
  2. trainer trains on the accumulated buffer and publishes current.pt
     (atomic rename -> the eval server reloads it)

Usage: run_loop.py [--run-dir az/data/run1] [--workers 4] [--games 8]
                   [--sims 200] [--batch 32] [--train-steps 2000]
                   [--iterations 100] [--value-target blend] [--lam 0.5]
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import socket


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


ROOT = Path(__file__).parent.parent
AZ = ROOT / "chushogi-az"
PY = "/data2/cs/venv/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="az/data/run1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games", type=int, default=8, help="per worker per iter")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train-steps", type=int, default=2000)
    ap.add_argument("--train-batch", type=int, default=256)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--value-target", default="blend")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    run = Path(a.run_dir)
    games_dir = run / "games"
    ckpt_dir = run / "ckpt"
    games_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    current = ckpt_dir / "current.pt"

    port = free_port()
    server = subprocess.Popen(
        [PY, str(ROOT / "az/eval_server.py"), "--port", str(port),
         "--checkpoint", str(current), "--watch"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = open(run / "loop.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    try:
        # wait for the server
        deadline = time.time() + 120
        while time.time() < deadline:
            if "serving" in server.stdout.readline():
                break
        say(f"[loop] eval server on :{port}")

        step0 = 0
        for it in range(a.iterations):
            t0 = time.time()
            procs = []
            for w in range(a.workers):
                p = subprocess.Popen([str(AZ)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True)
                seed = a.seed + it * 100003 + w * 101
                p.stdin.write(
                    f"selfplay --games {a.games} --sims {a.sims} "
                    f"--batch {a.batch} --seed {seed} --out {games_dir} "
                    f"--eval pipe --port {port}\n")
                p.stdin.flush()
                procs.append(p)
            total_plies = 0
            total_games = 0
            for p in procs:
                for _ in range(a.games):
                    line = p.stdout.readline()
                    m = re.search(r"game \d+: (\d+) plies", line)
                    if m:
                        total_plies += int(m.group(1))
                        total_games += 1
                p.stdin.write("quit\n")
                p.stdin.flush()
                p.wait()
            say(f"[loop] iter {it}: {total_games} games, "
                f"avg plies {total_plies / max(total_games, 1):.0f}, "
                f"self-play {time.time() - t0:.0f}s")

            t0 = time.time()
            cmd = [PY, str(ROOT / "az/train.py"), "--games-dirs",
                   str(games_dir), "--out", str(current),
                   "--steps", str(a.train_steps), "--batch",
                   str(a.train_batch), "--step0", str(step0),
                   "--value-target", a.value_target, "--lam", str(a.lam)]
            if current.exists():
                cmd += ["--init-from", str(current)]
            t = subprocess.run(cmd, capture_output=True, text=True)
            if t.returncode != 0:
                say("[loop] trainer failed:\n" + t.stdout[-1500:] +
                    t.stderr[-1500:])
                return 1
            step0 += a.train_steps
            last = [x for x in t.stdout.splitlines() if " loss " in x]
            say(f"[loop] iter {it}: trained {a.train_steps} steps in "
                f"{time.time() - t0:.0f}s; {last[-1] if last else 'no log'}")
            # archive per-iteration checkpoints for A/B strength matches
            shutil.copy(current, ckpt_dir / f"iter_{it:04d}.pt")
        say("[loop] finished")
        return 0
    finally:
        server.terminate()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
