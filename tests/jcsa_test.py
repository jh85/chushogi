#!/usr/bin/env python3
"""JCSA rule-set tests: the spec cases of docs/jcsa/cases.tsv, each checked
under both rule sets (lishogi, the engine default, and `rules jcsa`), plus
extra cases for readings the spec rows do not exercise.

A case is a position plus either a move, with a legal / illegal verdict, or
"(status)", with the game-end verdict of the `status` command. A move written
"a->b (any route)" stands for any legal move from a to b. The spec gives no
winner; it is derived here: a stalemated side to move loses, and a bare
royal loses to the side with more pieces.

Usage: jcsa_test.py [--binary ./chushogi-gen] [--cases docs/jcsa/cases.tsv]
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_CASES = Path(__file__).resolve().parent.parent / "docs/jcsa/cases.tsv"

# (id, sfen, move or "(status)", verdict under lishogi, verdict under JCSA)
EXTRA = [
    # U2: the foot is judged with the capturing piece lifted, so the rook 5c
    # protects the Black lion 5h through the White rook 5f (X-ray).
    ("U2.xray-foot",
     "k10B/12/7R4/12/12/7r4/12/7N4/12/12/12/K11 w 1a 1",
     "5f5h", "illegal", "illegal"),
    ("U2.xray-foot-control",
     "k10B/12/12/12/12/7r4/12/7N4/12/12/12/K11 w 1a 1",
     "5f5h", "illegal", "legal"),
    # U3: two Black lions, judged separately: the lion 9e has a foot (gold
    # 9f), the promoted kirin 4e has none.
    ("U3.lion-with-foot",
     "k10B/12/12/12/3N1r2+O3/3G8/12/12/12/12/12/K11 w 1a 1",
     "7e9e", "illegal", "illegal"),
    ("U3.lion-without-foot",
     "k10B/12/12/12/3N1r2+O3/3G8/12/12/12/12/12/K11 w 1a 1",
     "7e4e", "illegal", "legal"),
    # R-L4 binds every non-lion, the king included: JCSA lets it take the
    # lion 6f without a foot, but not with one (gold 6e).
    ("R-L4.king-no-foot",
     "k11/12/12/12/4r7/6n5/6K5/12/12/12/12/12 b 5e 2",
     "6g6f", "illegal", "legal"),
    ("R-L4.king-foot",
     "k11/12/12/12/4r1g5/6n5/6K5/12/12/12/12/12 b 5e 2",
     "6g6f", "illegal", "illegal"),
    # U1: a lion may always take an adjacent lion, even a protected one
    # (vertical mover 4a) during the ban, on a double move's mid square too.
    ("U1.adjacent-lion-igui",
     "k7V2R/9n2/8N3/12/12/12/12/12/12/12/12/K11 w 1a 1",
     "3b4c3b", "legal", "legal"),
    ("U1.adjacent-lion-through",
     "k7V2R/9n2/8N3/12/12/12/12/12/12/12/12/K11 w 1a 1",
     "3b4c5d", "legal", "legal"),
    # A horned falcon's jump or second step onto a lion without a foot.
    ("R-L4.falcon-jump-no-foot",
     "k11/12/12/12/6P5/12/12/5+h6/12/5N6/12/11K w 6e 2",
     "7h7j", "illegal", "legal"),
    ("R-L4.falcon-second-step-no-foot",
     "k11/12/12/12/6P5/12/12/5+h6/12/5N6/12/11K w 6e 2",
     "7h7i7j", "illegal", "legal"),
    # Row 2 with a horned falcon: the gold 7j gives the Black lion 7i a foot.
    ("row2.falcon-step",
     "k11/12/12/12/6P5/12/12/5+h6/5N6/5G6/12/11K w 6e 2",
     "7h7i", "illegal", "illegal"),
    ("row2.falcon-igui",
     "k11/12/12/12/6P5/12/12/5+h6/5N6/5G6/12/11K w 6e 2",
     "7h7i7h", "legal", "illegal"),
    ("row2.falcon-through",
     "k11/12/12/12/6P5/12/12/5+h6/5N6/5G6/12/11K w 6e 2",
     "7h7i7j", "legal", "illegal"),
    # U4, row 1b: the lion on the capture square itself (a kirin took a lion
    # there and promoted). lishogi exempts that square; under JCSA the gold
    # 7j gives it a foot, so it may not be taken, on the mid square either.
    ("U4.capture-square-step",
     "k11/12/12/12/12/12/12/5+h6/5+O6/5G6/12/11K w 7i 2",
     "7h7i", "legal", "illegal"),
    ("U4.capture-square-igui",
     "k11/12/12/12/12/12/12/5+h6/5+O6/5G6/12/11K w 7i 2",
     "7h7i7h", "legal", "illegal"),
    ("U4.capture-square-no-foot",
     "k11/12/12/12/12/12/12/5+h6/5+O6/12/12/11K w 7i 2",
     "7h7i7h", "legal", "legal"),
    # R-P4 removes only the lance's quiet promotion inside the zone: entering
    # the zone (R-P1) or capturing in it (R-P2) still promotes on the last
    # rank. Gote mirror of C06.
    ("R-P4.lance-enters",
     "k11/12/12/12/2L9/12/12/12/12/12/12/K11 b - 1",
     "10e10a+", "legal", "legal"),
    ("R-P4.lance-captures",
     "k1g9/12/2L9/12/12/12/12/12/12/12/12/K11 b - 1",
     "10c10a+", "legal", "legal"),
    ("R-P4.lance-gote",
     "k11/12/12/12/12/12/12/12/12/12/9l2/K11 w - 1",
     "3k3l+", "legal", "illegal"),
    ("R-P4.lance-gote-defer",
     "k11/12/12/12/12/12/12/12/12/12/9l2/K11 w - 1",
     "3k3l", "legal", "legal"),
    # R-E4: king + one piece against a lone king wins under both; against
    # king + two pieces JCSA plays on. lishogi's counting is kept: the
    # winner's unpromoted pawn does not count.
    ("R-E4.one-piece",
     "12/12/12/12/6k5/4G7/6K5/12/12/12/12/12 w - 1",
     "(status)", "bareking", "bareking"),
    ("R-E4.two-pieces",
     "12/12/12/12/6k5/4G7/6K5/2S9/12/12/12/12 w - 1",
     "(status)", "bareking", "playing"),
    ("R-E4.pawn-not-counted",
     "12/12/12/12/6k5/4G7/6K5/2P9/12/12/12/12 w - 1",
     "(status)", "bareking", "bareking"),
]

SQUARE = re.compile(r"\d+[a-l]")
ANY_ROUTE = re.compile(r"(\d+[a-l])->(\d+[a-l]) \(any route\)")


class Engine:
    def __init__(self, binary, rules):
        self.binary = binary
        self.rules = rules
        self.start()

    def start(self):
        self.proc = subprocess.Popen(
            [self.binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        # lishogi runs on the default, so the default itself is tested.
        if self.rules != "lishogi":
            self.proc.stdin.write(f"rules {self.rules}\n")
            self.proc.stdin.flush()
            if self.proc.stdout.readline().rstrip("\n") != "ok":
                raise RuntimeError(f"engine rejected 'rules {self.rules}'")

    def query(self, line):
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
            out = self.proc.stdout.readline()
            if not out:
                raise BrokenPipeError
            return out.rstrip("\n")
        except (BrokenPipeError, ValueError):
            err = self.proc.stderr.read().strip()
            self.start()  # engine exited (e.g. rejected position): respawn
            raise RuntimeError(f"engine died on {line!r}: {err}")

    def close(self):
        self.proc.stdin.write("quit\n")
        self.proc.stdin.flush()
        self.proc.wait()


def expected_status(verdict, sfen):
    """The status line for a spec verdict such as "draw (jishogi)"."""
    word = verdict.split()[0]
    board, turn = sfen.split()[:2]
    if word == "stalemate":
        return f"status stalemate {'gote' if turn == 'b' else 'sente'}"
    if word == "bareking":
        sente = sum(c.isupper() for c in board)
        gote = sum(c.islower() for c in board)
        return f"status bareking {'sente' if sente > gote else 'gote'}"
    return f"status {word}"


def verdict_of(eng, sfen, move):
    """What the engine says: "legal"/"illegal", or the status line."""
    if eng.query(f"position sfen {sfen}") != "ok":
        raise RuntimeError("position rejected")
    if move == "(status)":
        return eng.query("status")
    legal = eng.query("legal").split()[1:]
    route = ANY_ROUTE.fullmatch(move)
    if route:
        a, b = route.groups()
        hit = any(SQUARE.findall(m)[0] == a and SQUARE.findall(m)[-1] == b
                  for m in legal)
    else:
        hit = move in legal
    return "legal" if hit else "illegal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./chushogi-gen")
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    a = ap.parse_args()

    with open(a.cases, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    cases = [(r["id"], r["sfen"], r["move"], r["verdict_lishogi"],
              r["verdict_jcsa"]) for r in rows] + EXTRA
    print(f"{len(rows)} spec cases ({a.cases}) + {len(EXTRA)} extra cases")

    failures = 0
    total = 0
    for column, rules in ((3, "lishogi"), (4, "jcsa")):
        eng = Engine(a.binary, rules)
        for case in cases:
            name, sfen, move = case[:3]
            want = case[column]
            if move == "(status)":
                want = expected_status(want, sfen)
            total += 1
            try:
                got = verdict_of(eng, sfen, move)
            except RuntimeError as e:
                got = f"error: {e}"
            if got == want:
                print(f"PASS {rules:7} {name}: {move} -> {got}")
            else:
                failures += 1
                print(f"FAIL {rules:7} {name}: {move}: got {got!r}, "
                      f"want {want!r}")
        eng.close()

    print(f"\n{total - failures}/{total} JCSA test cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
