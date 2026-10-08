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
                   [--eval-servers N] [--resume] [--python PATH]
"""

import argparse
import re
import shutil
import subprocess
import sys
import threading
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="az/data/run1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-servers", type=int, default=1,
                    help="eval server processes sharing the GPU; workers "
                         "are sharded round-robin across them")
    ap.add_argument("--games", type=int, default=8, help="per worker per iter")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train-steps", type=int, default=2000)
    ap.add_argument("--train-batch", type=int, default=256)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--value-target", default="blend")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--lr-decay-steps", type=int, default=0,
                    help="passthrough to train.py: cosine horizon in total "
                         "steps (0 = constant 1e-4 after warmup)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for eval_server.py / train.py")
    ap.add_argument("--resume", action="store_true",
                    help="continue iteration numbering after existing "
                         "iter_*.pt checkpoints in the run dir")
    ap.add_argument("--mate-nodes", type=int, default=5000,
                    help="root mate-probe budget per root (0=off)")
    ap.add_argument("--probe-depth", type=int, default=3,
                    help="in-tree shallow mate probe depth (0=off)")
    ap.add_argument("--probe-nodes", type=int, default=2000)
    a = ap.parse_args()

    run = Path(a.run_dir)
    games_dir = run / "games"
    ckpt_dir = run / "ckpt"
    games_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    current = ckpt_dir / "current.pt"

    log = open(run / "loop.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    def start_server(port):
        p = subprocess.Popen(
            [a.python, str(ROOT / "az/eval_server.py"), "--port", str(port),
             "--checkpoint", str(current), "--watch"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ready = threading.Event()

        def drain():  # keep stdout drained so the server never blocks on it
            for line in p.stdout:
                log.write(line)
                log.flush()
                if "serving" in line:
                    ready.set()

        threading.Thread(target=drain, daemon=True).start()
        return p, ready

    servers = []
    for _ in range(a.eval_servers):
        port = free_port()
        servers.append((port,) + start_server(port))

    try:
        deadline = time.time() + 120
        for _, _, ready in servers:
            ready.wait(max(deadline - time.time(), 0.0))
        say(f"[loop] {len(servers)} eval server(s) on "
            + ", ".join(f":{port}" for port, _, _ in servers))

        it0 = 0
        if a.resume:
            existing = sorted(ckpt_dir.glob("iter_*.pt"))
            if existing:
                it0 = int(existing[-1].stem.split("_")[1]) + 1
            say(f"[loop] resuming {current} at iter {it0}")

        step0 = 0
        for it in range(it0, it0 + a.iterations):
            t0 = time.time()
            procs = []
            for w in range(a.workers):
                port = servers[w % len(servers)][0]
                p = subprocess.Popen([str(AZ)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True)
                seed = a.seed + it * 100003 + w * 101
                p.stdin.write(
                    f"selfplay --games {a.games} --sims {a.sims} "
                    f"--batch {a.batch} --seed {seed} --out {games_dir} "
                    f"--eval pipe --port {port} --mate-nodes {a.mate_nodes} "
                    f"--probe-depth {a.probe_depth} --probe-nodes "
                    f"{a.probe_nodes}\n")
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
            cmd = [a.python, str(ROOT / "az/train.py"), "--games-dirs",
                   str(games_dir), "--out", str(current),
                   "--steps", str(a.train_steps), "--batch",
                   str(a.train_batch), "--step0", str(step0),
                   "--value-target", a.value_target, "--lam", str(a.lam)]
            if a.lr_decay_steps:
                cmd += ["--lr-decay-steps", str(a.lr_decay_steps)]
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
        for _, p, _ in servers:
            p.terminate()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
