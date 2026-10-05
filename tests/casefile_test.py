#!/usr/bin/env python3
"""Verify a (input, output) test-case file produced by make_cases.py.

For every line `position sfen ... [moves ...]\\tbestmove <m>`:
  - the input column must be a well-formed USI position command,
  - the output move must be accepted by the engine (it is in the generated
    legal move list for the input position),
  - consecutive cases must chain: within one game each input extends the
    previous one by exactly the previously played move; a non-chaining line
    must start a new game (no `moves` part).

Cases from one game share prefixes, so the engine is driven incrementally
(move + legal per case) and the whole file is checked in one linear pass.

Usage: casefile_test.py <cases.tsv> [--binary ./chushogi-gen]
"""

import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", type=str)
    ap.add_argument("--binary", default="./chushogi-gen")
    a = ap.parse_args()

    proc = subprocess.Popen([a.binary], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)

    def cmd(line, expect_prefix):
        proc.stdin.write(line + "\n")
        proc.stdin.flush()
        out = proc.stdout.readline()
        if not out:
            err = proc.stderr.read()
            raise RuntimeError(f"engine died on {line!r}: {err.strip()}")
        out = out.rstrip("\n")
        prefix, _, rest = out.partition(" ")
        if prefix != expect_prefix:
            raise RuntimeError(f"{line!r} -> {out!r}")
        return rest

    failures = 0
    total = 0
    prev_input = None
    prev_move = None

    with open(a.cases) as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            total += 1
            parts = line.split("\t")
            ok = len(parts) == 2
            inp, out = parts if ok else (None, None)
            out_move = None
            if ok:
                if not out.startswith("bestmove ") or len(out.split()) != 2:
                    ok = False
                else:
                    out_move = out.split()[1]
                if not inp.startswith("position sfen "):
                    ok = False
            if not ok:
                print(f"line {lineno}: malformed test case: {line!r}")
                failures += 1
                prev_input = None
                continue

            try:
                chained = False
                if prev_input is not None and prev_move is not None:
                    for ext in (f"{prev_input} {prev_move}",
                                f"{prev_input} moves {prev_move}"):
                        if inp == ext:
                            cmd(f"move {prev_move}", "ok")
                            chained = True
                            break
                if not chained:
                    if " moves " in inp:
                        print(f"line {lineno}: case does not chain with the "
                              f"previous one and is not a game start")
                        failures += 1
                    cmd(inp, "ok")  # engine validates the whole prefix
                legal = cmd("legal", "legal").split()
                if out_move not in legal:
                    print(f"line {lineno}: output {out_move} not in legal "
                          f"list ({len(legal)} moves)\n  input: {inp}")
                    failures += 1
                    prev_input = None
                    continue
            except RuntimeError as e:
                print(f"line {lineno}: {e}")
                failures += 1
                prev_input = None
                continue
            prev_input, prev_move = inp, out_move

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait()
    print(f"{total - failures}/{total} test cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
