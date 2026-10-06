#!/usr/bin/env python3
"""AZ strength evaluation: matches between an AZ player (chushogi-az `go az`,
driven by an eval server on --port) and an opponent — the random player
(chushogi-gen `go random`) or another AZ instance.

Referee protocol mirrors tools/random_match.py: each side receives the start
position + all moves so far and answers with its move; a third engine
instance validates moves and adjudicates the game end (status, fourfold
repetition, 1000-ply cap).

Usage: eval_match.py --port 51589 [--games 20] [--sims 200]
       [--opponent random|az2] [--port2 N] [--seed 1]
Prints the score from the AZ player's perspective.
"""

import argparse
import subprocess
import sys

INITIAL = ("lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/"
           "3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1")
PLY_CAP = 1000


class Player:
    def __init__(self, binary, go, seed=None, port=None):
        self.proc = subprocess.Popen([binary], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True)
        self.go = go
        if seed is not None:
            self.cmd(f"seed {seed}", "ok")
        if port is not None:
            self.cmd(f"eval-server --port {port}", "ok")

    def cmd(self, line, expect):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = self.proc.stdout.readline().rstrip("\n")
        head, _, rest = out.partition(" ")
        if head != expect:
            raise RuntimeError(f"{line!r} -> {out!r}")
        return rest

    def move(self, initial, moves):
        self.cmd(f"position sfen {initial}" +
                 (" moves " + " ".join(moves) if moves else ""), "ok")
        return self.cmd(self.go, "bestmove")

    def close(self):
        self.proc.stdin.write("quit\n")
        self.proc.stdin.flush()
        self.proc.wait(timeout=10)


def play_game(az, opp, ref, az_color, seed_base):
    """az_color: 0 = az plays sente. Returns 1/0.5/0 from AZ's perspective."""
    players = [az, opp] if az_color == 0 else [opp, az]
    ref.cmd(f"position sfen {INITIAL}", "ok")
    moves = []
    seen = {}
    while True:
        side = len(moves) % 2
        best = players[side].move(INITIAL, moves)
        if best == "resign":
            winner = 1 - side
            return (1.0 if winner == az_color else 0.0), "resign"
        ref.cmd(f"move {best}", "ok")
        moves.append(best)
        sfen = ref.cmd("sfen", "sfen")
        key = " ".join(sfen.split()[:3])
        seen[key] = seen.get(key, 0) + 1
        status = ref.cmd("status", "status").split()
        if status[0] != "playing":
            if len(status) > 1:  # someone won
                winner = 0 if status[1] == "sente" else 1
                return (1.0 if winner == az_color else 0.0), status[0]
            return 0.5, status[0]
        if seen[key] >= 4:
            return 0.5, "repetition"
        if len(moves) >= PLY_CAP:
            return 0.5, "moveLimit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--az", default="./chushogi-az")
    ap.add_argument("--gen", default="./chushogi-gen")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--port2", type=int, default=None)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mate-nodes", type=int, default=0)
    ap.add_argument("--probe-depth", type=int, default=0)
    a = ap.parse_args()

    extra = ""
    if a.mate_nodes:
        extra += f" --mate-nodes {a.mate_nodes}"
    if a.probe_depth:
        extra += f" --probe-depth {a.probe_depth}"

    az = Player(a.az, f"go az --sims {a.sims}{extra}", port=a.port)
    if a.port2 is not None:
        opp = Player(a.az, f"go az --sims {a.sims}{extra}", port=a.port2)
        opp_name = f"az@{a.port2}"
    else:
        opp = Player(a.gen, "go random", seed=a.seed)
        opp_name = "random"
    ref = Player(a.gen, "go random")  # referee; never asked for moves

    score = 0.0
    counts = {}
    try:
        for g in range(a.games):
            az_color = g % 2
            pts, reason = play_game(az, opp, ref, az_color, a.seed + g)
            score += pts
            counts[reason] = counts.get(reason, 0) + 1
            print(f"game {g}: {'win' if pts == 1 else 'draw' if pts else 'loss'}"
                  f" ({reason})", flush=True)
    finally:
        for p in (az, opp, ref):
            p.close()
    n = a.games
    print(f"AZ vs {opp_name}: {score}/{n} = {100 * score / n:.1f}%  {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
