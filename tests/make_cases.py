#!/usr/bin/env python3
"""Build a (input, output) test-case file from downloaded lishogi chu shogi
game records.

Format: one test case per line, two tab-separated columns, both in USI
protocol style:
  column 1 (input):  position sfen <board> <turn> <lioncap|-> <num> [moves m1 ... mk]
                     the game's start position plus the ordered move sequence
                     up to the current position (a literal USI 'position'
                     command)
  column 2 (output): bestmove <mk+1>
                     the move the side-to-move player selected (a literal USI
                     'bestmove' response)

Every ply of every game produces one case: N plies -> N cases.

Usage: make_cases.py <output.tsv> <games_dir> [games_dir ...]
"""

import argparse
import sys
from pathlib import Path


def cases_from_game(gdir):
    sfens = (gdir / "positions.sfen").read_text().splitlines()
    moves = (gdir / "moves.usi").read_text().split()
    if len(sfens) != len(moves) + 1:
        raise ValueError(f"{gdir.name}: sfen/move count mismatch")
    initial = sfens[0]
    for i in range(len(moves)):
        inp = f"position sfen {initial}"
        if i:
            inp += " moves " + " ".join(moves[:i])
        yield inp, f"bestmove {moves[i]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("games_dirs", type=Path, nargs="+")
    a = ap.parse_args()

    games = []
    for d in a.games_dirs:
        games.extend(p for p in d.iterdir()
                     if p.is_dir() and (p / "positions.sfen").exists())
    games.sort(key=lambda p: p.name)
    if not games:
        print("no games found")
        return 1

    seen = set()
    n_cases = 0
    with a.output.open("w") as f:
        for gdir in games:
            for inp, out in cases_from_game(gdir):
                f.write(f"{inp}\t{out}\n")
                seen.add(inp)
                n_cases += 1
    print(f"{a.output}: {n_cases} test cases from {len(games)} games "
          f"({len(seen)} distinct inputs, {n_cases - len(seen)} shared prefixes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
