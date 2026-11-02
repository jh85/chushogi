#!/usr/bin/env python3
"""Compare end-to-end searches with isolated inference servers and saved histories."""
import argparse
import concurrent.futures
import json
import hashlib
import queue
import threading
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from eval_match import Player

ROOT=Path(__file__).resolve().parent.parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--positions',required=True,type=Path)
    ap.add_argument('--az',required=True)
    ap.add_argument('--server-script',default=str(ROOT/'az/eval_server.py'))
    ap.add_argument('--eval-policy', choices=['full','legal'], default='legal')
    ap.add_argument('--profile-eval',action='store_true')
    ap.add_argument('--sims',type=int,default=256)
    ap.add_argument('--root-policy-temp',type=float,default=1.25)
    ap.add_argument('--mate-nodes',type=int,default=5000)
    ap.add_argument('--probe-depth',type=int,default=3)
    ap.add_argument('--probe-nodes',type=int,default=2000)
    ap.add_argument('--max-batch',type=int,default=0)
    ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--servers',type=int,default=1)
    ap.add_argument('--repeats',type=int,default=1)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    os.environ["CHUSHOGI_EVAL_POLICY"]=a.eval_policy
    if min(a.servers,a.repeats,a.workers)<=0:ap.error('counts must be positive')
    positions=json.loads(a.positions.read_text())
    if not positions:ap.error('positions file is empty')
    a.out.mkdir(parents=True,exist_ok=False)
    ports=[];procs=[]
    try:
        for server in range(a.servers):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            ports.append(port)
            cmd=[sys.executable,a.server_script,'--checkpoint',a.checkpoint,'--port',str(port)]
            if a.max_batch:cmd+=['--max-batch',str(a.max_batch),'--batch-wait-ms','1','--tcp-nodelay']
            if a.profile_eval:cmd.append('--profile-eval')
            log_path=a.out/f'server_{server}.log'
            with open(log_path,'w') as log:
                procs.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(ROOT/'az')}))
        deadline=time.monotonic()+120
        for i,proc in enumerate(procs):
            while 'serving' not in (a.out/f'server_{i}.log').read_text():
                if proc.poll() is not None or time.monotonic()>deadline:raise RuntimeError((a.out/f'server_{i}.log').read_text())
                time.sleep(.1)
        trials=[]
        for repeat in range(a.repeats):
            jobs=queue.Queue()
            for p in positions:jobs.put(p)
            workers=min(a.workers,len(positions))
            # Persistent clients remove connection startup from search timing.
            barrier=threading.Barrier(workers+1)
            def worker(slot):
                player=Player(a.az,'',port=ports[slot%len(ports)])
                results=[]
                try:
                    p=positions[slot%len(positions)]
                    player.cmd('position sfen '+p['initial']+' moves '+' '.join(p['moves']),'ok')
                    player.cmd('go az --sims 16 --batch 16 --root-policy-temp 1.25','bestmove')
                    barrier.wait(timeout=120)
                    while True:
                        try:p=jobs.get_nowait()
                        except queue.Empty:break
                        player.cmd('seed 20261021','ok')
                        player.cmd('position sfen '+p['initial']+' moves '+' '.join(p['moves']),'ok')
                        start=time.monotonic()
                        r=json.loads(player.cmd(f'analyze az --sims {a.sims} --batch 16 --root-policy-temp {a.root_policy_temp} '
                            f'--mate-nodes {a.mate_nodes} --probe-depth {a.probe_depth} --probe-nodes {a.probe_nodes}','analysis'))
                        r.update(id=p['id'],seconds=time.monotonic()-start)
                        results.append(r)
                    return results
                except BaseException:
                    barrier.abort()
                    raise
                finally:player.close()
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures=[pool.submit(worker,slot) for slot in range(workers)]
                barrier.wait(timeout=120)
                start=time.monotonic()
                results=[r for f in futures for r in f.result()]
                elapsed=time.monotonic()-start
            assert len(results)==len(positions)
            trial=dict(seconds=elapsed,searches=len(results),searches_per_second=len(results)/elapsed,
                       results=sorted(results,key=lambda r:r['id']))
            trials.append(trial)
            print(json.dumps({k:v for k,v in trial.items() if k!='results'}),flush=True)
        report=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},trials=trials,
                    checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                    positions_sha256=hashlib.sha256(a.positions.read_bytes()).hexdigest())
        if a.repeats==1:report.update(trials[0])
        (a.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        for proc in procs:proc.terminate()
        for proc in procs:proc.wait(timeout=10)

if __name__=='__main__':main()
