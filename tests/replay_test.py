#!/usr/bin/env python3
"""Replay lishogi chu shogi game records against the C++ move generator.

Every position of every game is a test case. For each ply of each game:
  1. the engine is fed the initial SFEN plus the ordered USI moves so far
     (statefully: set once, then one move at a time),
  2. the recorded next move must appear in the generated legal-move list
     (moves are decoded to USI by the engine's conversion layer),
  3. after applying the move, the engine's SFEN must equal the recorded SFEN
     exactly (board, side to move, lion-capture field, move number).

Usage: replay_test.py <games_dir> [--binary ./chushogi-gen]
Exit code 0 iff all test cases pass.
"""

import argparse
import subprocess
import sys
from pathlib import Path


class Engine:
    def __init__(self, binary):
        self.proc = subprocess.Popen(
            [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        self.game = "?"

    def cmd(self, line, expect):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = self.proc.stdout.readline()
        if not out:
            err = self.proc.stderr.read()
            raise RuntimeError(
                f"engine died on {line!r} (game {self.game}): {err.strip()}")
        out = out.rstrip("\n")
        if expect == "ok":
            if out != "ok":
                raise RuntimeError(f"game {self.game}: {line!r} -> {out!r}")
            return None
        prefix, _, rest = out.partition(" ")
        if prefix != expect:
            raise RuntimeError(f"game {self.game}: {line!r} -> {out!r}")
        return rest

    def close(self):
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def test_game(gdir, binary, verbose):
    sfens = (gdir / "positions.sfen").read_text().splitlines()
    moves = (gdir / "moves.usi").read_text().split()
    if len(sfens) != len(moves) + 1:
        return False, f"sfen/move count mismatch: {len(sfens)} vs {len(moves)}"

    eng = Engine(binary)
    eng.game = gdir.name
    failures = []
    tested = 0
    try:
        eng.cmd(f"position sfen {sfens[0]}", "ok")
        got = eng.cmd("sfen", "sfen")
        if got != sfens[0]:
            failures.append(f"initial sfen mismatch:\n  got {got}\n  exp {sfens[0]}")
        for i, mv in enumerate(moves):
            legal = eng.cmd("legal", "legal")
            legal_moves = legal.split() if legal else []
            tested += 1
            if mv not in legal_moves:
                failures.append(
                    f"ply {i + 1}: played move {mv} not in legal list "
                    f"({len(legal_moves)} moves)\n  sfen {sfens[i]}")
                break
            eng.cmd(f"move {mv}", "ok")
            got = eng.cmd("sfen", "sfen")
            if got != sfens[i + 1]:
                failures.append(
                    f"ply {i + 1} ({mv}): sfen mismatch\n  got {got}\n  exp {sfens[i + 1]}")
                break
    except RuntimeError as e:
        failures.append(str(e))
    finally:
        eng.close()
    if verbose and not failures:
        return True, f"{len(moves)} plies, {tested} positions OK"
    return not failures, "; ".join(failures) if failures else f"{len(moves)} plies OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games_dir", type=Path)
    ap.add_argument("--binary", default="./chushogi-gen")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    games = sorted(p for p in a.games_dir.iterdir()
                   if p.is_dir() and (p / "positions.sfen").exists())
    if not games:
        print(f"no games found under {a.games_dir}")
        return 1

    total_plies = 0
    failed = 0
    for gdir in games:
        ok, msg = test_game(gdir, a.binary, a.verbose)
        plies = len((gdir / "moves.usi").read_text().split())
        total_plies += plies
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"{status} {gdir.name}: {plies} plies" + ("" if ok and not a.verbose else f"  {msg}"))
    print(f"\n{len(games) - failed}/{len(games)} games passed, "
          f"{total_plies} positions tested each as (initial SFEN + ordered moves)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
