#!/usr/bin/env python3
"""AZ pipe-eval end-to-end test (milestone 3): the Python eval server
(az/eval_server.py, untrained net) drives a C++ self-play game; the record
is validated like the M2 smoke test."""

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--az", default="./chushogi-az")
    ap.add_argument("--gen", default="./chushogi-gen")
    ap.add_argument("--py", default="/data2/cs/venv/bin/python")
    a = ap.parse_args()

    port = free_port()
    server = subprocess.Popen(
        [a.py, "az/eval_server.py", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        # wait for the server banner
        deadline = time.time() + 120
        while time.time() < deadline:
            line = server.stdout.readline()
            if "serving" in line:
                break
            if server.poll() is not None:
                print("FAIL: eval server died:", line)
                return 1
        else:
            print("FAIL: eval server did not start")
            return 1

        tmp = Path(tempfile.mkdtemp(prefix="azpipe_"))
        worker = subprocess.Popen(
            [a.az], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        worker.stdin.write(
            f"selfplay --games 1 --sims 16 --batch 8 --seed 5 "
            f"--out {tmp} --eval pipe --port {port}\n")
        worker.stdin.flush()
        banner = worker.stdout.readline()
        if not banner.startswith("game 0"):
            print("FAIL: worker:", banner)
            return 1
        worker.stdin.write("quit\n")
        worker.stdin.flush()
        worker.wait(timeout=60)

        recs = list(tmp.glob("game_*.json"))
        if len(recs) != 1:
            print("FAIL: no record written")
            return 1
        rec = json.loads(recs[0].read_text())
        if not rec["samples"] or not rec["moves"]:
            print("FAIL: empty game")
            return 1
        for smp in rec["samples"][:: max(1, len(rec["samples"]) // 10)]:
            probs = [p for _, p in smp["policy"]]
            if abs(sum(probs) - 1.0) > 1e-4:
                print(f"FAIL: policy sums to {sum(probs)}")
                return 1
        # moves replay through the main engine
        gen = subprocess.Popen([a.gen], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, text=True)
        gen.stdin.write("position sfen " + rec["initial"] + " moves " +
                        " ".join(rec["moves"]) + "\n")
        gen.stdin.flush()
        if gen.stdout.readline().strip() != "ok":
            print("FAIL: moves not accepted by the engine")
            return 1
        gen.stdin.write("quit\n")
        gen.stdin.flush()
        gen.wait(timeout=30)
        print(f"1/1 az pipe-eval game ok ({len(rec['moves'])} plies, "
              f"{rec['reason']})")
        return 0
    finally:
        server.terminate()


if __name__ == "__main__":
    sys.exit(main())
