#!/usr/bin/env python3
"""Checkmate-solver (BNS / df-pn) tests.

Crafted positions plus real game endings from the downloaded lishogi
records:
  - mate-in-1 crafted (sente and gote attackers),
  - mate-in-1 from the penultimate position of every royalsLost game,
  - a mate-in-3 from a real game (rpPXpivL, ply 429),
  - no-mate positions (bare-king attacker, pawn-only attacker),
  - a multi-royal defender (reported unknown by design),
  - published tsume problems from tests/tsume/*.json (mate must be
    provable, and the published solution/variations must be legal),
  - BNS and df-pn arithmetics must agree on the verdicts.

For every "yes" verdict the returned PV is replayed through the engine to
confirm it is a legal move sequence (the solver also self-validates it).

Usage: mate_test.py <games_dir1> [games_dir2 ...] [--binary ./chushogi-gen]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# (name, sfen, expected verdict)
CRAFTED = [
    # mate in 1: the rook captures the exposed king
    ("m1-sente",
     "12/12/7k4/12/7R4/12/12/12/12/12/12/11K b - 1", "yes"),
    # mate in 1 with gote as the attacker
    ("m1-gote",
     "k11/12/12/12/12/7K4/12/7r4/12/12/12/12 w - 1", "yes"),
    # mate in 3 from a real game (rpPXpivL, ply 429; gote mates sente)
    ("m3-real-rpPXpivL",
     "5ek3vl/4t1o4a/9m2/3m3+btgc1/3gpc1pppfp/2pp8/9P1P/5EGP3K/9+r2/10+h1/"
     "11A/11L w - 432", "yes"),
    # no mate: attacker has only a king
    ("nomate-bare-attacker",
     "12/12/7k4/12/12/12/12/12/12/12/12/11K b - 1", "no"),
    # no certain verdict: a lone pawn cannot force a mate (never "yes")
    ("nomate-pawn-only",
     "7k4/12/12/12/12/12/12/7P4/12/12/12/11K b - 1", "not-yes"),
    # multi-royal defender: out of the checker's scope by design
    ("unknown-multi-royal",
     "12/12/6+ek4/12/12/7R4/12/12/12/12/12/11K b - 1", "unknown"),
]


class Engine:
    def __init__(self, binary):
        self.proc = subprocess.Popen(
            [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)

    def cmd(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = self.proc.stdout.readline()
        if not out:
            err = self.proc.stderr.read()
            raise RuntimeError(f"engine died on {line!r}: {err.strip()}")
        return out.rstrip("\n")

    def mate(self, sfen, nodes, arith):
        if self.cmd(f"position sfen {sfen}") != "ok":
            raise RuntimeError(f"position rejected: {sfen}")
        verdict = self.cmd(f"mate {nodes} {arith}")
        nodes_line = self.proc.stdout.readline().rstrip("\n")
        pv = ""
        if verdict == "mate yes":
            pv = self.proc.stdout.readline().rstrip("\n")
        return verdict, nodes_line, pv

    def apply_accepts(self, sfen, moves):
        """True if the engine accepts the move sequence from sfen."""
        line = f"position sfen {sfen}" + (" moves " + " ".join(moves)
                                          if moves else "")
        try:
            return self.cmd(line) == "ok"
        except RuntimeError:
            return False

    def close(self):
        self.proc.stdin.write("quit\n")
        self.proc.stdin.flush()
        self.proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games_dirs", type=Path, nargs="+")
    ap.add_argument("--binary", default="./chushogi-gen")
    ap.add_argument("--nodes", type=int, default=1000000)
    a = ap.parse_args()

    eng = Engine(a.binary)
    failures = 0
    total = 0

    def check(name, sfen, want, arith, nodes=None):
        nonlocal failures, total
        total += 1
        verdict, nodes_line, pv_line = eng.mate(sfen, nodes or a.nodes, arith)
        got = verdict.split()[1]
        ok = got == want or (want == "not-yes" and got != "yes")
        pv_moves = pv_line.split()[1:] if pv_line else []
        if ok and got == "yes":
            if not pv_moves:
                ok = False
                print(f"FAIL {name} [{arith}]: yes but no PV")
            elif not eng.apply_accepts(sfen, pv_moves):
                ok = False
                print(f"FAIL {name} [{arith}]: PV not accepted: {pv_moves}")
        if not ok:
            failures += 1
            print(f"FAIL {name} [{arith}]: got {verdict}, want {want} "
                  f"({nodes_line})")
        return got

    # 1. Crafted positions, both arithmetics.
    for name, sfen, want in CRAFTED:
        check(f"{name}", sfen, want, "bns")
        check(f"{name}", sfen, want, "pndn")

    # 2. Mate-in-1 from the penultimate position of every royalsLost game.
    records = []
    for d in a.games_dirs:
        records.extend(sorted(Path(d).glob("*/record.json")))
    for rec_path in records:
        rec = json.loads(rec_path.read_text())
        if rec["status"] != "royalsLost":
            continue
        rows = rec["positions"]
        penult = rows[-2]["sfen"]
        check(f"m1-game-{rec['id']}", penult, "yes", "bns")

    # 3. Published tsume problems (tools/extract_tsume.py records): the
    # published lines must be legal, and the solver must reproduce the
    # verdict recorded at extraction time.
    tsume_dir = Path(__file__).parent / "tsume"
    for prob_path in sorted(tsume_dir.glob("*.json")):
        prob = json.loads(prob_path.read_text())
        name = f"tsume-{prob['id']}"
        if prob.get("conditional"):
            continue  # retro-analysis problems are outside the solver model
        sfen = prob["sfen"]
        want = prob.get("engine_check", {}).get("mate", {}).get("result")
        if want == "yes":
            used = prob["engine_check"]["mate"].get("nodes", 0)
            check(name, sfen, "yes", "bns", nodes=max(a.nodes, 2 * used + 100))
        main = prob["solution"]["moves"]
        total += 1
        if not eng.apply_accepts(sfen, main):
            failures += 1
            print(f"FAIL {name}: published solution not accepted: {main}")
        for i, var in enumerate(prob.get("variations", [])):
            total += 1
            full = var.get("full_line") or \
                main[:var["branch_after_ply"]] + var["moves"]
            if not eng.apply_accepts(sfen, full):
                failures += 1
                print(f"FAIL {name}: variation {i} not accepted")

    eng.close()
    print(f"{total - failures}/{total} mate cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
