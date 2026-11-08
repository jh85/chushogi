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
import fcntl
import math
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import socket
from selfplay_pool import run_dynamic


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
    ap.add_argument("--az", default=str(AZ), help="self-play/search engine binary")
    ap.add_argument("--init-from", default="",
                    help="seed a new run from this checkpoint with a fresh replay directory")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-servers", type=int, default=1,
                    help="eval server processes sharing the GPU; workers "
                         "are sharded round-robin across them")
    ap.add_argument("--games", type=int, default=8, help="per worker per iter")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train-steps", type=int, default=2000)
    ap.add_argument("--train-batch", type=int, default=256)
    ap.add_argument("--max-positions", type=int, default=1_000_000,
                    help="newest replay positions retained by the trainer")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--strength-every", type=int, default=0,
                    help="run paired checkpoint matches every N iterations (0=off)")
    ap.add_argument("--strength-pairs", type=int, default=20)
    ap.add_argument("--strength-sims", type=int, default=256)
    ap.add_argument("--strength-workers", type=int, default=4)
    ap.add_argument("--value-target", default="blend")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--lr-decay-steps", type=int, default=0,
                    help="passthrough to train.py: cosine horizon in total "
                         "steps (0 = constant 1e-4 after warmup)")
    ap.add_argument("--lr", type=float, default=0,
                    help="passthrough to train.py peak LR (0 = its default)")
    ap.add_argument("--optimizer", choices=["adamw", "muon_adamw"], default="adamw")
    ap.add_argument("--muon-lr-scale", type=float, default=1.)
    ap.add_argument("--muon-warmup-steps", type=int, default=50)
    ap.add_argument("--safe-sampling", action="store_true", help="experimental exclusion of proved-lost/hanging root moves")
    ap.add_argument("--safe-policy", action="store_true", help="experimental zeroing of unsafe root policy targets")
    ap.add_argument("--dynamic-games", action="store_true", help="assign individual games to free workers")
    ap.add_argument("--eval-max-batch", type=int, default=0)
    ap.add_argument("--eval-batch-wait-ms", type=float, default=1.0)
    ap.add_argument("--eval-tcp-nodelay", action="store_true")
    ap.add_argument("--temp-plies", type=int, default=20, help="sample first N plies; -1 samples throughout")
    ap.add_argument("--root-policy-temp", type=float, default=1.0)
    ap.add_argument("--train-lock", default="", help="shared file lock to serialize GPU training across branches")
    ap.add_argument("--stop-file", default="", help="finish the current iteration and stop when this file exists")
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
    if min(a.workers, a.eval_servers, a.games, a.train_steps,
           a.train_batch, a.max_positions, a.strength_pairs,
           a.strength_sims, a.strength_workers) <= 0 or a.strength_every < 0:
        ap.error("counts must be positive; --strength-every may be zero")
    if not math.isfinite(a.root_policy_temp) or a.root_policy_temp <= 0:
        ap.error("root policy temperature must be finite and positive")
    if a.eval_max_batch < 0 or not math.isfinite(a.eval_batch_wait_ms) or a.eval_batch_wait_ms < 0:
        ap.error("inference batching settings must be finite and nonnegative")
    if a.temp_plies < -1:
        ap.error("--temp-plies must be -1 or nonnegative")
    if not math.isfinite(a.muon_lr_scale) or a.muon_lr_scale <= 0 or a.muon_warmup_steps < 0:
        ap.error("invalid Muon scale/warmup")
    a.az = str(Path(a.az).resolve())

    run = Path(a.run_dir).resolve()
    games_dir = run / "games"
    ckpt_dir = run / "ckpt"
    games_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    current = ckpt_dir / "current.pt"
    if current.exists() and not a.resume:
        ap.error("run already has current.pt; use --resume or a new --run-dir")
    if a.init_from:
        if current.exists():
            ap.error("--init-from requires a new run without current.pt")
        shutil.copyfile(a.init_from, current)
        shutil.copyfile(a.init_from, ckpt_dir / "anchor.pt")
    (run / "loop_config.json").write_text(json.dumps(vars(a), indent=2) + "\n")

    log = open(run / "loop.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    def start_server(port):
        extra = ["--max-batch", str(a.eval_max_batch), "--batch-wait-ms", str(a.eval_batch_wait_ms)]
        if a.eval_tcp_nodelay:
            extra.append("--tcp-nodelay")
        p = subprocess.Popen(
            [a.python, str(ROOT / "az/eval_server.py"), "--port", str(port),
             "--checkpoint", str(current), "--watch", *extra],
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

        for it in range(it0, it0 + a.iterations):
            if a.stop_file and Path(a.stop_file).exists():
                say("[loop] stopped at completed iteration boundary")
                break
            t0 = time.time()
            safety_options = ((" --safe-sampling 1" if a.safe_sampling else "") +
                              (" --safe-policy 1" if a.safe_policy else ""))
            if a.dynamic_games:
                options=(f"--temp-plies {a.temp_plies} --sims {a.sims} --batch {a.batch} --root-policy-temp {a.root_policy_temp} "
                         f"--mate-nodes {a.mate_nodes} --probe-depth {a.probe_depth} --probe-nodes {a.probe_nodes}{safety_options}")
                total_plies,total_games=run_dynamic(a.az,[s[0] for s in servers],a.workers,a.games,
                    a.seed,it,games_dir/f"iter_{it:06d}",options)
            else:
                procs = []
                for w in range(a.workers):
                    # Isolate iterations and workers: PID-based record names can
                    # otherwise overwrite old games after PID reuse.
                    worker_dir = games_dir / f"iter_{it:06d}" / f"worker_{w:04d}"
                    port = servers[w % len(servers)][0]
                    p = subprocess.Popen([a.az], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, text=True)
                    seed = a.seed + it * 100003 + w * 101
                    p.stdin.write(
                        f"selfplay --games {a.games} --sims {a.sims} "
                        f"--temp-plies {a.temp_plies} --root-policy-temp {a.root_policy_temp} --batch {a.batch} --seed {seed} --out {worker_dir} "
                        f"--eval pipe --port {port} --mate-nodes {a.mate_nodes} "
                        f"--probe-depth {a.probe_depth} --probe-nodes "
                        f"{a.probe_nodes}{safety_options}\n")
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
            if total_games != a.workers*a.games:
                raise RuntimeError(f"incomplete self-play iteration: {total_games} games")
            say(f"[loop] iter {it}: {total_games} games, "
                f"avg plies {total_plies / max(total_games, 1):.0f}, "
                f"self-play {time.time() - t0:.0f}s")

            t0 = time.time()
            cmd = [a.python, str(ROOT / "az/train.py"), "--games-dirs",
                   str(games_dir), "--out", str(current),
                   "--steps", str(a.train_steps), "--batch",
                   str(a.train_batch), "--seed", str(a.seed + it * 100003),
                   "--max-positions", str(a.max_positions),
                   "--value-target", a.value_target, "--lam", str(a.lam),
                   "--optimizer", a.optimizer, "--muon-lr-scale", str(a.muon_lr_scale),
                   "--muon-warmup-steps", str(a.muon_warmup_steps)]
            if a.lr_decay_steps:
                cmd += ["--lr-decay-steps", str(a.lr_decay_steps)]
            if a.lr:
                cmd += ["--lr", str(a.lr)]
            if current.exists():
                cmd += ["--init-from", str(current)]
            if a.train_lock:
                with open(a.train_lock, "a") as lock:
                    say("[loop] waiting for GPU training lock")
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    t = subprocess.run(cmd, capture_output=True, text=True)
            else:
                t = subprocess.run(cmd, capture_output=True, text=True)
            (run / f"train_{it:06d}.log").write_text(t.stdout + t.stderr)
            if t.returncode != 0:
                say("[loop] trainer failed:\n" + t.stdout[-1500:] +
                    t.stderr[-1500:])
                return 1
            last = [x for x in t.stdout.splitlines() if " loss " in x]
            say(f"[loop] iter {it}: trained {a.train_steps} steps in "
                f"{time.time() - t0:.0f}s; {last[-1] if last else 'no log'}")
            # archive per-iteration checkpoints for A/B strength matches
            shutil.copy(current, ckpt_dir / f"iter_{it:04d}.pt")
            if a.strength_every and (it + 1) % a.strength_every == 0:
                anchor = ckpt_dir / "anchor.pt"
                if not anchor.exists():
                    shutil.copyfile(ckpt_dir / "iter_0000.pt", anchor)
                previous = ckpt_dir / f"iter_{max(0, it-a.strength_every):04d}.pt"
                candidates = [str(anchor)]
                if previous.exists():
                    candidates.append(str(previous))
                candidates.append(str(ckpt_dir / f"iter_{it:04d}.pt"))
                result_dir = run / "strength" / f"iter_{it:06d}"
                result_dir.parent.mkdir(exist_ok=True)
                with open(run / f"strength_{it:06d}.log", "w") as strength_log:
                    result = subprocess.run(
                        [a.python, str(ROOT / "az/checkpoint_tournament.py"),
                         "--checkpoints", *candidates, "--pairs", str(a.strength_pairs),
                         "--sims", str(a.strength_sims), "--workers",
                         str(a.strength_workers), "--az", a.az,
                         "--mate-nodes", str(a.mate_nodes), "--probe-depth", str(a.probe_depth),
                         "--probe-nodes", str(a.probe_nodes),
                         "--eval-max-batch", str(a.eval_max_batch),
                         "--eval-batch-wait-ms", str(a.eval_batch_wait_ms),
                         *(["--eval-tcp-nodelay"] if a.eval_tcp_nodelay else []),
                         "--out", str(result_dir)],
                        stdout=strength_log, stderr=subprocess.STDOUT, cwd=ROOT)
                if result.returncode:
                    say(f"[loop] strength evaluation failed; see strength_{it:06d}.log")
                    return 1
                say(f"[loop] strength results: {result_dir / 'summary.json'}")
        say("[loop] finished")
        return 0
    finally:
        for _, p, _ in servers:
            p.terminate()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
