#!/usr/bin/env python3
"""Random-player chu shogi match referee.

Two random players (separate engine processes) play games against each other.
They communicate strictly USI-style: on every turn the referee sends the
side-to-move player the start position plus ALL moves played so far
(`position sfen <initial> moves ...`), and the player answers with its choice
(`go random` -> `bestmove <usi>`). The referee (a third engine instance)
validates each move, detects the game end, and writes a game record in the
same SFEN/USI style as the downloaded lishogi records:

  <output>/games/<id>/moves.usi       space-separated USI moves
  <output>/games/<id>/positions.sfen  one SFEN per line, initial first
  <output>/games/<id>/positions.tsv   ply, usi, sfen
  <output>/games/<id>/record.json     metadata incl. result and seeds

Game-end handling mirrors lishogi: royals lost, bare king, stalemate,
insufficient-material draw, fourfold repetition (draw; the perpetual-check
penalty is NOT implemented - random players do not exploit it), and the
1000-ply server cap (draw).

Usage: random_match.py [--games N] [--seed N] [--binary ./chushogi-gen]
                       [--output DIR] [--initial-sfen SFEN]
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

INITIAL = ("lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/"
           "3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1")
PLY_CAP = 1000


class Engine:
    def __init__(self, binary, seed=None):
        self.proc = subprocess.Popen(
            [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        if seed is not None:
            self.cmd(f"seed {seed}", "ok")

    def cmd(self, line, expect):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = self.proc.stdout.readline()
        if not out:
            err = self.proc.stderr.read()
            raise RuntimeError(f"engine died on {line!r}: {err.strip()}")
        out = out.rstrip("\n")
        head, _, rest = out.partition(" ")
        if head != expect:
            raise RuntimeError(f"{line!r} -> {out!r}")
        return rest

    def close(self):
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def play_game(binary, game_id, initial_sfen, seed_sente, seed_gote):
    sente = Engine(binary, seed_sente)
    gote = Engine(binary, seed_gote)
    ref = Engine(binary)
    players = [sente, gote]
    try:
        ref.cmd(f"position sfen {initial_sfen}", "ok")
        sfens = [ref.cmd("sfen", "sfen")]
        seen = {" ".join(sfens[0].split()[:3]): 1}
        moves = []
        result = None
        while result is None:
            side = len(moves) % 2
            player = players[side]
            player.cmd(f"position sfen {initial_sfen}" +
                       (" moves " + " ".join(moves) if moves else ""), "ok")
            best = player.cmd("go random", "bestmove")
            if best == "resign":
                result = {"status": "stalemate",
                          "winner": ["gote", "sente"][side],
                          "end_reason": "stalemate"}
                break
            ref.cmd(f"move {best}", "ok")  # validates the move
            moves.append(best)
            sfen = ref.cmd("sfen", "sfen")
            sfens.append(sfen)
            key = " ".join(sfen.split()[:3])
            seen[key] = seen.get(key, 0) + 1

            status = ref.cmd("status", "status").split()
            if status[0] != "playing":
                name = status[0]
                result = {
                    "status": {"bareking": "bareKing",
                               "royalslost": "royalsLost",
                               "stalemate": "stalemate",
                               "draw": "draw"}[name],
                    "winner": status[1] if len(status) > 1 else None,
                    "end_reason": name,
                }
            elif seen[key] >= 4:
                result = {"status": "draw", "winner": None,
                          "end_reason": "repetition"}
            elif len(moves) >= PLY_CAP:
                result = {"status": "draw", "winner": None,
                          "end_reason": "moveLimit"}
        return moves, sfens, result
    finally:
        for p in players:
            p.close()
        ref.close()


def write_record(outdir, game_id, initial_sfen, moves, sfens, result,
                 seed_sente, seed_gote):
    gdir = Path(outdir) / "games" / game_id
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "moves.usi").write_text(" ".join(moves) + "\n")
    (gdir / "positions.sfen").write_text("\n".join(sfens) + "\n")
    with (gdir / "positions.tsv").open("w") as f:
        f.write("ply\tusi\tsfen\n")
        for ply, sfen in enumerate(sfens):
            usi = moves[ply - 1] if ply else ""
            f.write(f"{ply}\t{usi}\t{sfen}\n")
    record = {
        "id": game_id,
        "variant": "chushogi",
        "rated": False,
        "created_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat().replace("+00:00", "Z"),
        "status": result["status"],
        "winner": result["winner"],
        "end_reason": result["end_reason"],
        "players": {
            "sente": {"name": f"random-seed-{seed_sente}"},
            "gote": {"name": f"random-seed-{seed_gote}"},
        },
        "plies": len(moves),
        "initial_sfen": initial_sfen,
        "final_sfen": sfens[-1],
        "moves_usi": moves,
    }
    (gdir / "record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--binary", default="./chushogi-gen")
    ap.add_argument("--output", default="random_games")
    ap.add_argument("--initial-sfen", default=INITIAL)
    ap.add_argument("--id-prefix", default="rr")
    a = ap.parse_args()

    outdir = Path(a.output)
    for i in range(a.games):
        game_id = f"{a.id_prefix}{i + 1:06d}"
        seed_s, seed_g = a.seed + 2 * i, a.seed + 2 * i + 1
        moves, sfens, result = play_game(
            a.binary, game_id, a.initial_sfen, seed_s, seed_g)
        write_record(outdir, game_id, a.initial_sfen, moves, sfens, result,
                     seed_s, seed_g)
        print(f"{game_id}: {len(moves)} plies, {result['status']}"
              f" ({result['end_reason']}), winner: {result['winner']}",
              flush=True)
    print(f"\n{a.games} game record(s) written under {outdir}/games/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
