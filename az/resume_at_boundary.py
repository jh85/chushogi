#!/usr/bin/env python3
"""Hand over a running loop at its next completed iteration, then resume it.

Does not terminate processes or interrupt games/training. A STOP marker makes
the old loop exit itself. On an abnormal exit, resume is refused.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--old-pid',type=int,required=True)
    ap.add_argument('--az',type=Path,required=True)
    ap.add_argument('--final-iteration',type=int,required=True)
    a=ap.parse_args();run=a.run_dir.resolve();engine=a.az.resolve(strict=True)
    cfg=json.loads((run/'loop_config.json').read_text())
    cmdline=Path(f'/proc/{a.old_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(run) not in cmdline or 'run_loop.py' not in cmdline:raise RuntimeError('old PID is not the specified training loop')
    marker=Path(cfg['stop_file'])
    state=run/'throughput_handover.json'
    if state.exists():raise RuntimeError('handover already exists')
    token=f'throughput handover {os.getpid()}\n'
    info=dict(status='waiting_for_iteration_boundary',old_pid=a.old_pid,engine=str(engine),
              target_final_iteration=a.final_iteration,requested_at=time.time())
    state.write_text(json.dumps(info,indent=2)+'\n')
    with marker.open('x') as f:f.write(token)
    while alive(a.old_pid):time.sleep(5)
    if (run/'loop.log').read_text().splitlines()[-1]!='[loop] finished':
        info['status']='old_loop_failed';state.write_text(json.dumps(info,indent=2)+'\n')
        raise RuntimeError('old loop did not finish cleanly; inspect logs, not resuming')
    checkpoints=sorted((run/'ckpt').glob('iter_*.pt'))
    last=int(checkpoints[-1].stem.split('_')[1])
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(run/'ckpt/current.pt')!=digest(checkpoints[-1]):raise RuntimeError('current checkpoint does not match completed iteration')
    remaining=a.final_iteration-last
    if remaining<=0:raise RuntimeError('requested final iteration already reached')
    if (run/'games'/f'iter_{last+1:06d}').exists():raise RuntimeError('next iteration already has game data')
    if marker.read_text()!=token:raise RuntimeError('STOP marker changed; refusing to override it')
    # The old loop only terminates servers in its finally block. Allow those
    # processes time to release their resources before allocating the new server.
    time.sleep(2)
    shutil.copy2(run/'loop_config.json',run/f'loop_config_before_throughput_{last:04d}.json')
    cfg.update(az=str(engine),iterations=remaining,eval_servers=1,eval_max_batch=128,
               eval_batch_wait_ms=1.0,eval_tcp_nodelay=True,dynamic_games=True,resume=True)
    command=[cfg['python'],str(Path(__file__).resolve().with_name('run_loop.py')),'--resume']
    for key,value in cfg.items():
        if key in ('resume','init_from'):continue
        option='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:command.append(option)
        else:command += [option,str(value)]
    marker.unlink()
    with open(run/'throughput_console.log','a') as log:
        proc=subprocess.Popen(command,cwd=Path(__file__).resolve().parent.parent,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    process_record=run/'continuation_process.json'
    if process_record.exists():
        shutil.copy2(process_record,run/f'continuation_process_before_throughput_{last:04d}.json')
    process_record.write_text(json.dumps(dict(pid=proc.pid,command=command,checkpoint=str(checkpoints[-1]),
        workers=cfg['workers'],games_per_iteration=cfg['workers']*cfg['games'],additional_iterations=remaining,
        target_final_iteration=a.final_iteration,throughput_optimized=True),indent=2)+'\n')
    info.update(status='resumed',new_pid=proc.pid,completed_iteration=last,
                remaining_iterations=remaining,command=command,resumed_at=time.time())
    state.write_text(json.dumps(info,indent=2)+'\n')


if __name__=='__main__':main()
