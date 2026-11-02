#!/usr/bin/env python3
"""Pause at an iteration boundary, benchmark server counts, then resume the winner.

Run as the account owning the training directory. Does not kill the training
loop. On failure, leaves the STOP marker in place and saves a diagnostic state.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def choose_winner(measurements):
    # Prefer fewer processes if the measured medians are exactly equal.
    return min(measurements,key=lambda n:(statistics.median(measurements[n]),n))


def resume_command(config, last, final, winner):
    options=dict(config)
    if last>=final:raise ValueError('training already reached the target iteration')
    options.update(iterations=final-last,eval_servers=winner)
    command=[options['python'],str(ROOT/'az/run_loop.py'),'--resume']
    for key,value in options.items():
        if key in ('resume','init_from'):continue
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:command.append(flag)
        else:command.extend([flag,str(value)])
    return command


def positions_from_iteration(run, iteration, count):
    files=sorted((run/'games'/f'iter_{iteration:06d}').rglob('game_*.json'))
    if not files:raise ValueError('no games for the last completed iteration')
    chosen=files if len(files)<=count else ([files[len(files)//2]] if count==1 else
        [files[round(i*(len(files)-1)/(count-1))] for i in range(count)])
    positions=[]
    for i,path in enumerate(chosen):
        record=json.loads(path.read_text())
        if len(record['moves'])<2:continue
        ply=max(1,min(len(record['moves'])-1,int(len(record['moves'])*[.15,.4,.65][i%3])))
        positions.append(dict(id=i,initial=record['initial'],moves=record['moves'][:ply],
                              source=str(path),ply=ply))
    if not positions:raise ValueError('no nonterminal histories available')
    return positions


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--old-pid',type=int,required=True)
    ap.add_argument('--final-iteration',type=int,default=139)
    ap.add_argument('--servers',type=int,nargs='+',default=[2,3,4])
    ap.add_argument('--rounds',type=int,default=4)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if min(*a.servers,a.rounds)<=0:ap.error('server counts and rounds must be positive')
    run=a.run_dir.resolve();out=a.out.resolve()
    if not os.access(run,os.W_OK):ap.error(f'{run} is not writable; run as its owner (currently root)')
    cfg=json.loads((run/'loop_config.json').read_text())
    tokens=Path(f'/proc/{a.old_pid}/cmdline').read_bytes().decode().split('\0')
    if '--run-dir' not in tokens or 'run_loop.py' not in ' '.join(tokens):ap.error('old PID is not a training loop')
    process_run=Path(tokens[tokens.index('--run-dir')+1])
    if not process_run.is_absolute():process_run=Path(os.readlink(f'/proc/{a.old_pid}/cwd'))/process_run
    if process_run.resolve()!=run:ap.error('old PID belongs to another run directory')
    marker=Path(cfg['stop_file'])
    if not marker.is_absolute():marker=ROOT/marker
    if marker.exists():ap.error('STOP file already exists; inspect the existing stop request first')
    out.mkdir(parents=True,exist_ok=False)
    state=dict(status='waiting_for_completed_iteration',old_pid=a.old_pid,run_dir=str(run),
               final_iteration=a.final_iteration,requested_at=time.time())
    def save():
        (out/'status.json').write_text(json.dumps(state,indent=2)+'\n')
        print(state['status'],flush=True)
    token=f'inference server tuning {os.getpid()}\n'
    save()
    try:
        with marker.open('x') as f:f.write(token)
        while alive(a.old_pid):time.sleep(5)
        if (run/'loop.log').read_text().splitlines()[-1]!='[loop] finished':
            raise RuntimeError('old loop did not finish cleanly; not resuming')
        # Allow terminated inference servers to release their GPU resources.
        time.sleep(3)
        last=max(int(p.stem.split('_')[1]) for p in (run/'ckpt').glob('iter_*.pt'))
        checkpoint=run/'ckpt'/f'iter_{last:04d}.pt'
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        if digest(checkpoint)!=digest(run/'ckpt/current.pt'):raise RuntimeError('archived/current checkpoint mismatch')
        if (run/'games'/f'iter_{last+1:06d}').exists():raise RuntimeError('next iteration already has game data')
        if last>=a.final_iteration:raise RuntimeError('target already reached; no resume needed')
        state.update(status='benchmarking',completed_iteration=last,checkpoint_sha256=digest(checkpoint));save()
        positions=positions_from_iteration(run,last,cfg['workers']*cfg['games'])
        (out/'positions.json').write_text(json.dumps(positions,indent=2)+'\n')
        measurements={n:[] for n in sorted(set(a.servers))};orders=[]
        for round_id in range(a.rounds):
            order=sorted(measurements,reverse=bool(round_id%2));orders.append(order)
            for count in order:
                trial=out/f'round_{round_id}_servers_{count}'
                command=[cfg['python'],str(ROOT/'az/benchmark_search.py'),'--checkpoint',str(checkpoint),
                    '--positions',str(out/'positions.json'),'--az',cfg['az'],'--workers',str(cfg['workers']),
                    '--servers',str(count),'--repeats','1','--max-batch',str(cfg['eval_max_batch']),'--out',str(trial)]
                with open(out/'benchmark.log','a') as log:
                    subprocess.run(command,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT,
                        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
                result=json.loads((trial/'results.json').read_text())
                records=result['trials'][0]['results']
                if len(records)!=len(positions) or len({r['id'] for r in records})!=len(positions):
                    raise RuntimeError('benchmark omitted or duplicated a position')
                measurements[count].append(result['trials'][0]['seconds'])
                state.update(measurements=measurements);save()
        winner=choose_winner(measurements)
        summary=dict(winner=winner,trial_orders=orders,measurements=measurements,
                     medians={n:statistics.median(v) for n,v in measurements.items()},
                     caveat='Search benchmark; full-iteration speedup remains to be measured.')
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        command=resume_command(cfg,last,a.final_iteration,winner)
        if marker.read_text()!=token:raise RuntimeError('STOP request changed; not overriding it')
        shutil.copy2(run/'loop_config.json',out/'previous_loop_config.json')
        marker.unlink()
        with open(run/'server_tuned_console.log','a') as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        state.update(status='resumed',new_pid=proc.pid,winner=winner,command=command,
                     remaining_iterations=a.final_iteration-last,resumed_at=time.time());save()
        print(f'Resumed PID {proc.pid} with {winner} servers, target iteration {a.final_iteration}.',flush=True)
    except BaseException as error:
        if not marker.exists():
            try:
                with marker.open('x') as f:f.write(token)
            except FileExistsError:pass
        state.update(status='failed',error=str(error));save()
        raise


if __name__=='__main__':main()
