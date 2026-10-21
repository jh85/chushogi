#!/usr/bin/env python3
"""Run reproducible checkpoint matches, pairing identical openings across colors.

Example: python az/checkpoint_tournament.py --checkpoints a.pt b.pt c.pt \
  --pairs 20 --sims 256 --out az/data/tournament
The later checkpoint in each pair is scored against the earlier one.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from eval_match import PLY_CAP, Player, play_game


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--pairs', type=int, default=20)
    ap.add_argument('--sims', type=int, default=256)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--opening-plies', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--az', default='./chushogi-az')
    ap.add_argument('--gen', default='./chushogi-gen')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if len(a.checkpoints) < 2 or min(a.pairs, a.sims, a.workers) <= 0:
        ap.error('at least two checkpoints and positive pairs/sims/workers required')
    if not 0 <= a.opening_plies < PLY_CAP:
        ap.error('opening plies must be between zero and the game ply cap')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    paths = [str(Path(p).resolve(strict=True)) for p in a.checkpoints]
    (out/'config.json').write_text(json.dumps(vars(a), indent=2)+'\n')
    servers, logs, ports = [], [], []
    try:
        for i, path in enumerate(paths):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            ports.append(port)
            log = open(out/f'server_{i}.log', 'w'); logs.append(log)
            servers.append(subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name('eval_server.py')),
                 '--port', str(port), '--checkpoint', path],
                stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}))
        deadline = time.monotonic()+120
        for i, server in enumerate(servers):
            while 'serving' not in (out/f'server_{i}.log').read_text():
                if server.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError(f'eval server failed: {out}/server_{i}.log')
                time.sleep(.2)

        def match(job):
            later, earlier, pair = job
            players = []
            try:
                for port in (ports[later], ports[earlier]):
                    players.append(Player(a.az, f'go az --sims {a.sims}', port=port))
                players.append(Player(a.gen, 'go random'))
                players.append(Player(a.gen, 'go random', seed=a.seed+pair))
                records = []
                for color in (0, 1):
                    players[3].cmd(f'seed {a.seed+pair}', 'ok')
                    moves = []
                    score, reason = play_game(*players[:3], color, a.seed+pair,
                                             a.opening_plies, players[3], moves)
                    record = dict(later=paths[later], earlier=paths[earlier],
                                  pair=pair, color=color, score=score,
                                  reason=reason, moves=moves)
                    (out/f'game_{later}_{earlier}_{pair}_{color}.json').write_text(
                        json.dumps(record, indent=2)+'\n')
                    print(f'{later} vs {earlier}, pair {pair}, color {color}: '
                          f'{score} ({reason}, {len(moves)} plies)', flush=True)
                    records.append(record)
                return records
            finally:
                for player in players:
                    player.close()

        jobs = [(j, i, pair) for pair in range(a.pairs)
                for j in range(len(paths)) for i in range(j)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            records = [r for result in pool.map(match, jobs) for r in result]
        summary = []
        for j in range(len(paths)):
            for i in range(j):
                scores = [r['score'] for r in records
                          if r['later']==paths[j] and r['earlier']==paths[i]]
                summary.append(dict(later=paths[j], earlier=paths[i],
                                    games=len(scores), wins=scores.count(1),
                                    draws=scores.count(.5), losses=scores.count(0),
                                    score=sum(scores)/len(scores)))
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2))
    finally:
        for server in servers:
            server.terminate()
        for server in servers:
            server.wait(timeout=10)
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
