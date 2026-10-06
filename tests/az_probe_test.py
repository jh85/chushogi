#!/usr/bin/env python3
"""Mate-probe tests (root override + in-tree shallow probes).

  1. Root override: on a mate-in-1 position, `go az --mate-nodes` returns the
     winning move with no search (even sims=1); the initial position (no
     mate) falls through to a normal move.
  2. In-tree probe: the king is under rook attack down an open file; stepping
     further along the file lets the rook capture next ply. With
     --probe-depth 1 the child is marked a proven loss, so the MCTS must
     choose a safe move (off the file).
"""

import subprocess
import sys

M1 = "12/6g5/7k4/12/12/12/12/12/12/12/12/7R3K b - 1"       # 5l5c wins
HANG = "10kr/12/12/12/12/12/12/12/12/12/12/10GK b - 1"     # 1l1k hangs the king
INITIAL = ("lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/"
           "3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1")


def main():
    az = subprocess.Popen(["./chushogi-az"], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True)

    def cmd(line):
        az.stdin.write(line + "\n")
        az.stdin.flush()
        return az.stdout.readline().strip()

    total = failures = 0

    def check(name, sfen, args, accept):
        nonlocal failures, total
        cmd(f"position sfen {sfen}")
        out = cmd(f"go az {args}").split(" ", 1)[1]
        total += 1
        if not accept(out):
            failures += 1
            print(f"FAIL {name}: got {out}")

    # root override fires even at sims=1
    check("m1-root", M1, "--sims 1 --batch 1 --eval random --mate-nodes 50000",
          lambda m: m == "5l5c")
    # no mate at the initial position: normal (legal) move, not the override
    check("initial-nomate", INITIAL,
          "--sims 32 --batch 8 --eval random --mate-nodes 50000",
          lambda m: m != "resign" and len(m) >= 4)
    # in-tree probe: must not step along the attacked file
    check("hang-avoided", HANG,
          "--sims 64 --batch 8 --eval random --probe-depth 1",
          lambda m: m != "1l1k")

    az.stdin.write("quit\n")
    az.stdin.flush()
    print(f"{total - failures}/{total} az probe checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
