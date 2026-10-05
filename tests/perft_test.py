#!/usr/bin/env python3
"""Exact legal-move-count (perft depth-1) tests.

cases file: one `sfen<TAB>count` per line (extracted from scalashogi's own
chu shogi test fixtures, so the counts are the lishogi reference values).
Also verifies the initial-position perfts at depth 1 (36) and depth 2 (1296).

Usage: perft_test.py <perft_cases.tsv> [--binary ./chushogi-gen]
"""

import argparse
import subprocess
import sys

INITIAL = ("lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/"
           "3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1")


class Engine:
    def __init__(self, binary):
        self.proc = subprocess.Popen(
            [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)

    def cmd(self, line, prefix):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = self.proc.stdout.readline()
        if not out:
            err = self.proc.stderr.read()
            raise RuntimeError(f"engine died on {line!r}: {err.strip()}")
        out = out.rstrip("\n")
        p, _, rest = out.partition(" ")
        if p != prefix:
            raise RuntimeError(f"{line!r} -> {out!r}")
        return rest

    def legal(self):
        out = self.cmd("legal", "legal")
        return out.split() if out else []

    def close(self):
        self.proc.stdin.write("quit\n")
        self.proc.stdin.flush()
        self.proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", type=str)
    ap.add_argument("--binary", default="./chushogi-gen")
    a = ap.parse_args()

    eng = Engine(a.binary)
    failures = 0
    total = 0

    # Depth 1 and 2 from the initial position.
    eng.cmd(f"position sfen {INITIAL}", "ok")
    root = eng.legal()
    d1 = len(root)
    d2 = 0
    for mv in root:
        eng.cmd(f"position sfen {INITIAL} moves {mv}", "ok")
        d2 += len(eng.legal())
    for want, got, label in ((36, d1, "initial depth 1"),
                             (1296, d2, "initial depth 2")):
        total += 1
        if got != want:
            failures += 1
            print(f"FAIL {label}: got {got}, want {want}")

    for lineno, line in enumerate(open(a.cases), 1):
        total += 1
        sfen, want = line.rstrip("\n").split("\t")
        try:
            eng.cmd(f"position sfen {sfen}", "ok")
            got = len(eng.legal())
        except RuntimeError as e:
            print(f"FAIL line {lineno}: {e}")
            failures += 1
            continue
        if got != int(want):
            failures += 1
            print(f"FAIL line {lineno}: {sfen}\n  got {got}, want {want}")
    eng.close()
    print(f"{total - failures}/{total} perft cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
