#!/usr/bin/env python3
"""Wait for a fork pilot, summarize new self-play games, then evaluate checkpoints."""
import argparse
import base64
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def summarize(branch, start, end):
    games = []
    for iteration in range(start, end + 1):
        for path in (branch / 'games' / f'iter_{iteration:06d}').rglob('game_*.json'):
            games.append(json.loads(path.read_text()))
    material = [sum(b.bit_count() for b in base64.b64decode(g['samples'][-1]['planes'])[:78*144//8])
                for g in games if g['samples']]
    return dict(games=len(games), endings=dict(Counter(g['reason'] for g in games)),
                draws=sum(g['result']==0 for g in games),
                mean_plies=statistics.mean(len(g['moves']) for g in games) if games else None,
                mean_pieces_before_last_move=statistics.mean(material) if material else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', type=Path, required=True)
    a = ap.parse_args()
    out = a.experiment.resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    processes = json.loads((out / 'processes.json').read_text())
    status = out / 'pilot_status.json'
    status.write_text(json.dumps(dict(status='waiting_for_training', processes=processes), indent=2)+'\n')
    while any(alive(p['pid']) for p in processes):
        time.sleep(30)
    start = manifest['source_iteration'] + 1
    end = manifest['source_iteration'] + manifest['pilot_iterations']
    missing = [b['name'] for b in manifest['branches']
               if not (Path(b['run_dir']) / 'ckpt' / f'iter_{end:04d}.pt').exists()]
    if missing:
        status.write_text(json.dumps(dict(status='training_incomplete', missing_branches=missing), indent=2)+'\n')
        raise SystemExit('Pilot stopped before final checkpoints; no final evaluation launched.')
    summary = {b['name']:summarize(Path(b['run_dir']), start, end) for b in manifest['branches']}
    (out / 'selfplay_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    status.write_text(json.dumps(dict(status='evaluating', selfplay_summary=summary), indent=2)+'\n')
    command = json.loads((out / 'evaluation_command.json').read_text())
    with open(out / 'final_strength.log', 'a') as log:
        result = subprocess.run(command, cwd=Path(__file__).resolve().parent.parent,
                                stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'})
    status.write_text(json.dumps(dict(status='complete' if result.returncode==0 else 'evaluation_failed',
                                     evaluation_returncode=result.returncode), indent=2)+'\n')


if __name__ == '__main__':
    main()
