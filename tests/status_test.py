#!/usr/bin/env python3
"""Game-end status tests: bare king, insufficient material, royals lost,
stalemate — and the 1000-ply draw cap.

The crafted cases include scalashogi's own bare-king test positions. The
real-game checks evaluate the final position of downloaded games that ended
by bareKing / royalsLost and compare with the recorded outcome. The move cap
is a lishogi server rule (not position logic), so it is verified at the
record level: the one game that hit it sits at exactly 1000 plies, drawn.

Usage: status_test.py <games_dir1> [games_dir2 ...] [--binary ./chushogi-gen]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# (name, sfen, expected_status, expected_winner or None)
CRAFTED = [
    # K+G vs K: automatic win (scalashogi's own bare-king test position)
    ("bare-win-gold",
     "12/12/12/12/6k5/4G7/6K5/12/12/12/12/12 w - 1", "bareking", "sente"),
    # ...but not while the other side still has a piece
    ("not-bare-two-pieces",
     "12/12/12/12/6k5/4g7/4G7/6K5/12/12/12/12 b - 1", "playing", None),
    # ...and not while the lone king stands next to the gold (could capture)
    ("not-bare-adjacent-gold",
     "12/12/12/12/6k5/5G6/6K5/12/12/12/12/12 w - 1", "playing", None),
    # ...and not while the lone king checks the winner's king (scalashogi sit5)
    ("not-bare-kings-adjacent",
     "12/12/12/12/6k5/5K6/12/4G7/12/12/12/12 w - 1", "playing", None),
    # ...and not even with 3 winner pieces if the lone king gives check
    # (scalashogi sit7)
    ("not-bare-winner-in-check",
     "12/11G/12/12/3S8/12/7k4/6K5/12/12/12/12 w - 1", "playing", None),
    # a pawn does not count for the winning side (must promote safely first)
    ("pawn-does-not-win",
     "12/12/12/12/6k5/4P7/6K5/12/12/12/12/12 w - 1", "playing", None),
    # ...but a tokin (promoted pawn) does
    ("tokin-wins",
     "12/12/12/12/6k5/4+P7/6K5/12/12/12/12/12 w - 1", "bareking", "sente"),
    # a go-between does not count either...
    ("gb-does-not-win",
     "12/12/12/12/6k5/4I7/6K5/12/12/12/12/12 w - 1", "playing", None),
    # ...but the drunk elephant it promotes into does
    ("elephant-from-gb-wins",
     "12/12/12/12/6k5/4+I7/6K5/12/12/12/12/12 w - 1", "bareking", "sente"),
    # a dead pawn (stuck on the last rank) does not save the losing side
    ("dead-pawn-does-not-save",
     "12/12/12/12/6k5/4G7/6K5/12/12/12/12/7p4 w - 1", "bareking", "sente"),
    # ...while a living one does
    ("live-pawn-saves",
     "12/12/12/12/5pk5/4G7/6K5/12/12/12/12/12 w - 1", "playing", None),
    # insufficient material: bare king vs bare king is a draw...
    ("draw-kings-only",
     "12/12/12/12/6k5/12/12/6K5/12/12/12/12 b - 1", "draw", None),
    # (scalashogi sit3: dead pawns/lances on the back ranks are ignored)
    ("draw-dead-pieces-ignored",
     "1P3PP3P1/12/12/12/6k5/12/12/6K5/12/12/12/l10l b - 1", "draw", None),
    # ...K + dead pawn vs K is likewise a draw...
    ("draw-king-vs-king-dead-pawn",
     "7P4/12/12/12/6k5/12/12/6K5/12/12/12/12 w - 1", "draw", None),
    # ...as is baring the opponent right back (scalashogi sit4 after KxG)
    ("draw-bare-back",
     "12/12/12/12/12/5k6/12/5K6/12/12/12/12 b - 1", "draw", None),
    # ...but a go-between (or any other live piece) keeps the game alive
    # (scalashogi sit6: not bare, not draw)
    ("not-draw-gb-keeps-alive",
     "5P3P2/12/12/12/6k2i2/12/12/6K1I3/12/12/12/5p5p b - 1", "playing", None),
    # royals lost: the side to move has no royal left and loses
    ("royals-lost",
     "12/12/12/12/6g5/12/12/6K5/12/12/12/12 w - 1", "royalslost", "sente"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games_dirs", type=Path, nargs="+")
    ap.add_argument("--binary", default="./chushogi-gen")
    a = ap.parse_args()

    proc = subprocess.Popen([a.binary], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)

    def query(line):
        try:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
            out = proc.stdout.readline()
            if not out:
                raise BrokenPipeError
            return out.rstrip("\n")
        except (BrokenPipeError, ValueError):
            respawn()
            raise RuntimeError("engine died")

    def respawn():
        nonlocal proc
        proc = subprocess.Popen([a.binary], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1)

    def status_of(sfen):
        try:
            if query(f"position sfen {sfen}") != "ok":
                return None
            parts = query("status").split()
            return (parts[1], parts[2] if len(parts) > 2 else None)
        except RuntimeError:
            return None

    failures = 0
    total = 0

    # 1. Crafted positions.
    for name, sfen, want_status, want_winner in CRAFTED:
        total += 1
        got = status_of(sfen)
        if got != (want_status, want_winner):
            failures += 1
            print(f"FAIL {name}: got {got}, want {(want_status, want_winner)}")

    # 2. Real game endings: bareKing / royalsLost final positions must
    #    evaluate to the recorded outcome.
    records = []
    for d in a.games_dirs:
        records.extend(sorted(Path(d).glob("*/record.json")))
    for rec_path in records:
        rec = json.loads(rec_path.read_text())
        status = rec["status"]
        if status not in ("bareKing", "royalsLost"):
            continue
        total += 1
        final_sfen = rec["final_sfen"]
        want = ("bareking" if status == "bareKing" else "royalslost",
                rec["winner"])
        got = status_of(final_sfen)
        if got != want:
            failures += 1
            print(f"FAIL game {rec['id']}: final position got {got}, want {want}")

    # 3. The 1000-ply draw cap (server rule): exactly one game hit it.
    capped = [json.loads(p.read_text()) for p in records]
    capped = [r for r in capped if len(r["moves_usi"]) >= 1000]
    total += 1
    if not (len(capped) == 1 and len(capped[0]["moves_usi"]) == 1000
            and capped[0]["status"] == "draw" and capped[0]["winner"] is None):
        failures += 1
        print(f"FAIL move cap: expected exactly one 1000-ply draw, "
              f"got {[(r['id'], len(r['moves_usi']), r['status']) for r in capped]}")

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait()
    print(f"{total - failures}/{total} status cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
