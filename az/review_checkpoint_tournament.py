#!/usr/bin/env python3
"""Wait for the stronger tournament, report paired uncertainty, audit values."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--review-dir',type=Path,required=True)
    ap.add_argument('--wait-pid',type=int,required=True)
    a=ap.parse_args();root=a.review_dir.resolve();tournament=root/'tournament'
    status=root/'review_status.json'
    status.write_text(json.dumps(dict(status='waiting_for_tournament',pid=a.wait_pid),indent=2)+'\n')
    while alive(a.wait_pid):time.sleep(30)
    if not (tournament/'summary.json').exists():
        status.write_text(json.dumps(dict(status='tournament_incomplete'),indent=2)+'\n');raise SystemExit(1)
    summary=json.loads((tournament/'summary.json').read_text())
    cfg=json.loads((tournament/'config.json').read_text());paths=cfg['unique_checkpoints'];games=[]
    for path in tournament.glob('game_*.json'):games.append(json.loads(path.read_text()))
    expected=len(paths)*(len(paths)-1)//2*cfg['pairs']*2
    if len(games)!=expected:raise ValueError('incomplete tournament game count')
    total=defaultdict(float);counts=defaultdict(int);comparisons=[]
    for item in summary:
        later,earlier=item['later'],item['earlier'];pairs=defaultdict(list)
        for rec in games:
            if rec['later']==later and rec['earlier']==earlier:pairs[rec['pair']].append(rec)
        if len(pairs)!=cfg['pairs'] or any(len(v)!=2 or v[0]['moves'][:8]!=v[1]['moves'][:8] for v in pairs.values()):
            raise ValueError('opening/color pairing failed')
        means=np.array([np.mean([r['score'] for r in rows]) for rows in pairs.values()])
        rng=np.random.default_rng(20261029)
        ci=np.quantile(means[rng.integers(0,len(means),(100000,len(means)))].mean(axis=1),[.025,.975]).tolist()
        comparisons.append({**item,'paired_bootstrap_95pct':ci})
        total[later]+=item['score']*item['games'];counts[later]+=item['games']
        total[earlier]+=(1-item['score'])*item['games'];counts[earlier]+=item['games']
    ranking=sorted([dict(checkpoint=p,score=total[p]/counts[p],games=counts[p]) for p in paths],key=lambda r:-r['score'])
    final=next(c for c in comparisons if c['later']==paths[4] and c['earlier']==paths[3])
    regression=final['paired_bootstrap_95pct'][1]<.5
    report=dict(ranking=ranking,comparisons=comparisons,
        late_regression_confirmed_for_tested_matchup=regression,
        uncertainty='95% percentile bootstrap over paired openings; one training run, not independent training seeds.',
        causal_conclusion='Not established. Compare the training audit and value audit; a matched intervention is needed to identify a cause.')
    (root/'tournament_review.json').write_text(json.dumps(report,indent=2)+'\n')
    status.write_text(json.dumps(dict(status='auditing_values',late_regression_confirmed=regression),indent=2)+'\n')
    result=subprocess.run([sys.executable,str(Path(__file__).with_name('checkpoint_value_audit.py')),
        '--tournament',str(tournament),'--out',str(root/'value_audit.json')],
        cwd=Path(__file__).resolve().parent.parent,
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    status.write_text(json.dumps(dict(status='complete' if result.returncode==0 else 'value_audit_failed',
        late_regression_confirmed=regression),indent=2)+'\n')

if __name__=='__main__':main()
