#!/usr/bin/env python3
"""Run reproducible checkpoint matches, pairing identical openings across colors.

Example: python az/checkpoint_tournament.py --checkpoints a.pt b.pt c.pt \
  --pairs 20 --sims 256 --out az/data/tournament
The later checkpoint in each pair is scored against the earlier one.
"""
import argparse
import concurrent.futures
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from eval_match import PLY_CAP, Player, play_game


def unique_checkpoints(paths):
    """Keep the first path for each byte-identical checkpoint snapshot."""
    unique, seen = [], set()
    for path in paths:
        digest = hashlib.sha256()
        with open(path, 'rb') as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b''):
                digest.update(chunk)
        key = digest.digest()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def search_command(sims, mate_nodes=5000, probe_depth=3, probe_nodes=2000):
    return (f'go az --sims {sims} --mate-nodes {mate_nodes} '
            f'--probe-depth {probe_depth} --probe-nodes {probe_nodes}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--pairs', type=int, default=20)
    ap.add_argument('--sims', type=int, default=256)
    ap.add_argument('--mate-nodes', type=int, default=5000, help='root mate budget; 0 disables')
    ap.add_argument('--probe-depth', type=int, default=3, help='in-tree mate depth; 0 disables')
    ap.add_argument('--probe-nodes', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--eval-max-batch', type=int, default=0)
    ap.add_argument('--eval-batch-wait-ms', type=float, default=1.0)
    ap.add_argument('--eval-tcp-nodelay', action='store_true')
    ap.add_argument('--opening-plies', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--az', default='./chushogi-az')
    ap.add_argument('--gen', default='./chushogi-gen')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if len(a.checkpoints) < 2 or min(a.pairs, a.sims, a.workers) <= 0:
        ap.error('at least two checkpoints and positive pairs/sims/workers required')
    if min(a.mate_nodes, a.probe_depth) < 0 or a.probe_nodes <= 0:
        ap.error('mate/depth budgets must be nonnegative; probe-nodes positive')
    if not 0 <= a.opening_plies < PLY_CAP:
        ap.error('opening plies must be between zero and the game ply cap')
    if a.eval_max_batch < 0 or not math.isfinite(a.eval_batch_wait_ms) or a.eval_batch_wait_ms < 0:
        ap.error('inference batching settings must be finite and nonnegative')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    paths = [str(Path(p).resolve(strict=True)) for p in a.checkpoints]
    paths = unique_checkpoints(paths)
    (out/'config.json').write_text(json.dumps(
        {**vars(a), 'unique_checkpoints': paths}, indent=2)+'\n')
    if len(paths) < 2:
        (out/'summary.json').write_text('[]\n')
        print('Skipped strength matches: all supplied checkpoints are identical.', flush=True)
        return
    servers, logs, ports = [], [], []
    try:
        server_options = ['--max-batch', str(a.eval_max_batch), '--batch-wait-ms', str(a.eval_batch_wait_ms)]
        if a.eval_tcp_nodelay:
            server_options.append('--tcp-nodelay')
        for i, path in enumerate(paths):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            ports.append(port)
            log = open(out/f'server_{i}.log', 'w'); logs.append(log)
            servers.append(subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name('eval_server.py')),
                 '--port', str(port), '--checkpoint', path, *server_options],
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
                    players.append(Player(a.az, search_command(a.sims, a.mate_nodes, a.probe_depth, a.probe_nodes), port=port))
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
                games = [r for r in records
                         if r['later']==paths[j] and r['earlier']==paths[i]]
                scores = [r['score'] for r in games]
                summary.append(dict(later=paths[j], earlier=paths[i],
                                    games=len(scores), wins=scores.count(1),
                                    draws=scores.count(.5), losses=scores.count(0),
                                    score=sum(scores)/len(scores),
                                    endings=dict(Counter(r['reason'] for r in games)),
                                    mean_plies=sum(len(r['moves']) for r in games)/len(games)))
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
