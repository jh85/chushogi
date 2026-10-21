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
import json
import subprocess
import sys
from pathlib import Path

INITIAL = ("lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/"
           "3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1")
PLY_CAP = 1000


class Player:
    def __init__(self, binary, go, seed=None, port=None):
        self.proc = subprocess.Popen([binary], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True)
        self.go = go
        self.port = port  # set only for net-backed players (chushogi-az)
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


def play_game(az, opp, ref, az_color, seed_base, opening_plies=0, div=None,
              moves=None):
    """az_color: 0 = az plays sente. Returns 1/0.5/0 from AZ's perspective."""
    players = [az, opp] if az_color == 0 else [opp, az]
    # Reseed the net players per game so temperature sampling differs.
    for i, p in enumerate(dict.fromkeys(players)):
        if p.port is not None:
            p.cmd(f"seed {seed_base * 1009 + i}", "ok")
    ref.cmd(f"position sfen {INITIAL}", "ok")
    moves = [] if moves is None else moves
    seen = {" ".join(INITIAL.split()[:3]): 1}
    # Optional opening diversification: `div` plays random moves first, so
    # deterministic AZ-vs-AZ pairings don't replay one line every game.
    for _ in range(opening_plies):
        moves.append(div.move(INITIAL, moves))
        ref.cmd(f"move {moves[-1]}", "ok")
        sfen = ref.cmd("sfen", "sfen")
        key = " ".join(sfen.split()[:3])
        seen[key] = seen.get(key, 0) + 1
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
    ap.add_argument("--paired-openings", action="store_true",
                    help="reseed random openings identically for each color pair")
    ap.add_argument("--json-out", default="", help="save scores and full move traces")
    ap.add_argument("--mate-nodes", type=int, default=0)
    ap.add_argument("--probe-depth", type=int, default=0)
    ap.add_argument("--opening-plies", type=int, default=0,
                    help="random opening plies before the engines take over "
                         "(diversify deterministic AZ-vs-AZ matchups)")
    ap.add_argument("--temp-plies", type=int, default=0,
                    help="sample AZ moves from the visit distribution for "
                         "this many opening plies (noisy match play)")
    a = ap.parse_args()
    if a.games <= 0 or (a.paired_openings and a.games % 2):
        ap.error("games must be positive and even for paired openings")

    extra = ""
    if a.mate_nodes:
        extra += f" --mate-nodes {a.mate_nodes}"
    if a.probe_depth:
        extra += f" --probe-depth {a.probe_depth}"
    if a.temp_plies:
        extra += f" --temp-plies {a.temp_plies}"

    az = Player(a.az, f"go az --sims {a.sims}{extra}", port=a.port)
    if a.port2 is not None:
        opp = Player(a.az, f"go az --sims {a.sims}{extra}", port=a.port2)
        opp_name = f"az@{a.port2}"
    else:
        opp = Player(a.gen, "go random", seed=a.seed)
        opp_name = "random"
    ref = Player(a.gen, "go random")  # referee; never asked for moves
    div = (Player(a.gen, "go random", seed=a.seed * 1000003 + 7)
           if a.opening_plies else None)
    players = [p for p in (az, opp, ref, div) if p is not None]

    score = 0.0
    counts = {}
    records = []
    try:
        for g in range(a.games):
            az_color = g % 2
            if div and a.paired_openings:
                div.cmd(f"seed {a.seed * 1000003 + g // 2}", "ok")
            moves = []
            pts, reason = play_game(az, opp, ref, az_color, a.seed + g,
                                    a.opening_plies, div, moves)
            records.append(dict(game=g, az_color=az_color, score=pts,
                                reason=reason, moves=moves))
            if a.json_out:
                Path(a.json_out).write_text(json.dumps(
                    dict(config=vars(a), games=records), indent=2) + "\n")
            score += pts
            counts[reason] = counts.get(reason, 0) + 1
            print(f"game {g}: {'win' if pts == 1 else 'draw' if pts else 'loss'}"
                  f" ({reason})", flush=True)
    finally:
        for p in players:
            p.close()
    n = a.games
    print(f"AZ vs {opp_name}: {score}/{n} = {100 * score / n:.1f}%  {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
