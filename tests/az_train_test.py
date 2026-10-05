#!/usr/bin/env python3
"""AZ trainer smoke test (milestone 4): a few random-eval self-play games
feed the trainer; the checkpoint must load into the eval server and drive a
pipe-eval game."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from az_pipe_test import free_port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--az", default="./chushogi-az")
    ap.add_argument("--py", default="/data2/cs/venv/bin/python")
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="aztrain_"))
    games = tmp / "games"
    ckpt = tmp / "current.pt"

    # 1. random-eval games
    w = subprocess.Popen([a.az], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True)
    w.stdin.write(f"selfplay --games 3 --sims 8 --batch 4 --seed 11 "
                  f"--out {games}\nquit\n")
    w.stdin.flush()
    for _ in range(3):
        line = w.stdout.readline()
        assert line.startswith("game"), line
    w.wait(timeout=120)

    # 2. train briefly
    t = subprocess.run(
        [a.py, "az/train.py", "--games-dirs", str(games), "--out",
         str(ckpt), "--steps", "30", "--batch", "32", "--save-every", "15",
         "--log-every", "10", "--warmup-steps", "5"],
        capture_output=True, text=True, timeout=600)
    if t.returncode != 0:
        print(t.stdout[-2000:])
        print(t.stderr[-2000:])
        print("FAIL: trainer exited nonzero")
        return 1
    if not ckpt.exists():
        print("FAIL: no checkpoint")
        return 1
    losses = [float(x.split("loss ")[1].split(" ")[0])
              for x in t.stdout.splitlines() if " loss " in x]
    print(f"[test] trainer losses: {losses}")

    # 3. trained checkpoint drives a pipe game
    port = free_port()
    server = subprocess.Popen(
        [a.py, "az/eval_server.py", "--port", str(port), "--checkpoint",
         str(ckpt)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            line = server.stdout.readline()
            if "serving" in line:
                break
            if server.poll() is not None:
                print("FAIL: server died")
                return 1
        w = subprocess.Popen([a.az], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
        w.stdin.write(f"selfplay --games 1 --sims 8 --batch 4 --seed 13 "
                      f"--out {tmp}/g2 --eval pipe --port {port}\nquit\n")
        w.stdin.flush()
        ok = w.stdout.readline().startswith("game 0")
        w.wait(timeout=120)
        rec = json.loads(next((tmp / "g2").glob("game_*.json")).read_text())
        if not ok or not rec["moves"]:
            print("FAIL: trained-net game")
            return 1
        print(f"1/1 az train test ok (trained-net game: "
              f"{len(rec['moves'])} plies, {rec['reason']})")
        return 0
    finally:
        server.terminate()


if __name__ == "__main__":
    sys.exit(main())
