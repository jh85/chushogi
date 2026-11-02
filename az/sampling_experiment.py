#!/usr/bin/env python3
"""Prepare matched training forks: first-20-ply vs full-game move sampling.

Default: prepare only. --launch starts both loops and their final evaluator.
Never stops or modifies the source training run. Both branches retain the
source optimizer, replay window and training settings; root temperature=1.25.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from fork_training import prepare, launch, ROOT


def check_engine(binary):
    # Zero games probes the actual CLI without generating data or requiring GPU.
    with tempfile.TemporaryDirectory(prefix='sampling-engine-check-') as tmp:
        result=subprocess.run([str(binary)],input=f'selfplay --games 0 --temp-plies -1 --out {tmp}\nquit\n',
                              text=True,capture_output=True,timeout=10)
    if result.returncode:
        raise ValueError('engine does not accept --temp-plies; rebuild it: '+result.stderr.strip())


def prepare_sampling(source, iteration, out, binary, iterations=20, workers=22, games=4):
    return prepare(source,iteration,out,binary,iterations,
        branch_specs=[dict(name='opening20',root_policy_temp=1.25,temp_plies=20),
                      dict(name='all_moves',root_policy_temp=1.25,temp_plies=-1)],
        workers=workers,games=games,eval_servers=1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--iteration',type=int,required=True,help='completed iteration to snapshot')
    ap.add_argument('--az',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--iterations',type=int,default=20)
    ap.add_argument('--workers',type=int,default=22,help='per branch')
    ap.add_argument('--games',type=int,default=4,help='per worker; default gives 88 games per branch/iteration')
    ap.add_argument('--launch',action='store_true')
    a=ap.parse_args()
    if a.iteration<0 or min(a.iterations,a.workers,a.games)<=0:ap.error('counts must be positive; iteration nonnegative')
    check_engine(a.az.resolve(strict=True))
    manifest=prepare_sampling(a.source,a.iteration,a.out,a.az,a.iterations,a.workers,a.games)
    manifest.update(experiment='move_sampling',root_policy_temperature=1.25,
        sampling_temperature=1.0,games_per_iteration=a.workers*a.games,
        limitations=['One paired training seed; reduced draws do not establish improved strength.',
                     'Random streams diverge after different sampling decisions.'])
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Prepared opening20 and all_moves from iteration {a.iteration}, {a.iterations} additional iterations each.')
    print('Source training was not modified. Final evaluation uses greedy visit selection and root temperature 1.0.')
    if a.launch:
        print(json.dumps(launch(a.out),indent=2))
        with open(a.out/'watcher.log','a') as log:
            proc=subprocess.Popen([sys.executable,str(ROOT/'az/watch_training_fork.py'),
                '--experiment',str(a.out.resolve())],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        (a.out/'watcher_process.json').write_text(json.dumps(dict(pid=proc.pid),indent=2)+'\n')
        print('Launched final-evaluation watcher:',proc.pid)


if __name__=='__main__':main()
