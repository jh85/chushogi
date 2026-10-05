#!/usr/bin/env python3
"""MCTS sanity test (regression for the backup-perspective bug).

On a mate-in-1 position (the rook takes the hanging king), MCTS with a
uniform evaluator must choose the capturing move: the terminal edge must
out-prioritize every non-terminal alternative. Run for both colors (the
gote case exercises the canonical flip).
"""

import subprocess
import sys

SENTE_M1 = "12/6g5/7k4/12/12/12/12/12/12/12/12/7R3K b - 1"  # 5l5c wins
GOTE_M1 = "k6r4/12/12/12/12/12/12/12/12/7K4/6G5/12 w - 1"  # 5a5j wins


def main():
    az = subprocess.Popen(["./chushogi-az"], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True)

    def cmd(line):
        az.stdin.write(line + "\n")
        az.stdin.flush()
        return az.stdout.readline().strip()

    total = failures = 0
    for sfen, want in ((SENTE_M1, "5l5c"), (GOTE_M1, "5a5j")):
        cmd(f"position sfen {sfen}")
        out = cmd("go az --sims 64 --batch 8 --eval random")
        total += 1
        if out != f"bestmove {want}":
            failures += 1
            print(f"FAIL: {sfen} -> {out}, want bestmove {want}")
    az.stdin.write("quit\n")
    az.stdin.flush()
    print(f"{total - failures}/{total} az mcts checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
