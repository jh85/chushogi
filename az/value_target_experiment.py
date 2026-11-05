#!/usr/bin/env python3
"""Sequential, matched self-play continuations comparing z, softz and a0gb.

A common fresh self-play batch avoids stale A0GB labels from older binaries.
The baseline shares this same fresh replay start; this is a continuation pilot,
not a from-scratch comparison of the target algorithms.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def save(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def prepare_targets(source, iteration, out, binary, iterations=20,
                    workers=44, games=2, eval_servers=4, order_seed=20261105):
    from fork_training import prepare
    specs=[dict(name=mode,value_target=mode,sims=1024,root_policy_temp=1.25,
                temp_plies=-1,strength_every=0,dynamic_games=True,
                eval_max_batch=128,eval_batch_wait_ms=1.0,eval_tcp_nodelay=True)
           for mode in ('z','softz','a0gb')]
    manifest=prepare(source,iteration,out,binary,iterations,specs,workers,games,eval_servers)
    order=[b['name'] for b in manifest['branches']]
    random.Random(order_seed).shuffle(order)
    manifest.update(experiment='value_target',run_order=order,order_seed=order_seed,
        target_definitions=dict(z='Final game WDL, including explicit draws',
            softz='Search root mean score Q -> [Q,0,1-Q]',
            a0gb='Value at max-visit principal-variation endpoint, in root perspective -> [v,0,1-v]'),
        limitations=['One training seed and one branch order.',
            'All branches inherit weights and AdamW moments previously trained on z.',
            'All start with the same fresh replay batch, not the original historical replay.',
            'After updates, self-play trajectories and replay diverge.',
            'Soft targets encode draw score as a win/loss mixture; no explicit draw mass.',
            'Raw training losses across different target distributions are not strength comparisons.'])
    save(out/'manifest.json',manifest)
    return manifest


def bootstrap(protocol, out, state):
    from selfplay_pool import run_dynamic
    from fork_training import sha256
    source=Path(protocol['source'])
    common=out/'bootstrap'
    (common/'ckpt').mkdir(parents=True)
    checkpoint=common/'ckpt'/f"iter_{protocol['iteration']:04d}.pt"
    shutil.copy2(source/'ckpt'/checkpoint.name,checkpoint)
    if sha256(checkpoint)!=protocol['checkpoint_sha256']:
        raise RuntimeError('bootstrap checkpoint differs from selected source')
    cfg=json.loads((source/'loop_config.json').read_text())
    save(common/'loop_config.json',cfg)
    state.update(status='generating_common_replay',bootstrap_started_at=time.time(),
                 checkpoint_sha256=sha256(checkpoint))
    save(out/'status.json',state)
    servers=[];ports=[];logs=[]
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CHUSHOGI_EVAL_POLICY':'legal'}
    try:
        for i in range(4):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            ports.append(port)
            log=(common/f'server_{i}.log').open('w');logs.append(log)
            servers.append(subprocess.Popen([protocol['python'],str(out/'source/az/eval_server.py'),
                '--checkpoint',str(checkpoint),'--port',str(port),
                '--max-batch','128','--batch-wait-ms','1','--tcp-nodelay'],
                stdout=log,stderr=subprocess.STDOUT,env=env))
        deadline=time.monotonic()+120
        for i,proc in enumerate(servers):
            while 'serving' not in (common/f'server_{i}.log').read_text():
                if proc.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError(f'bootstrap eval server failed: {common}/server_{i}.log')
                time.sleep(.2)
        options='--temp-plies -1 --sims 1024 --batch 16 --root-policy-temp 1.25 --mate-nodes 5000 --probe-depth 3 --probe-nodes 2000'
        plies,games=run_dynamic(str(out/'bin/chushogi-az'),ports,44,2,20261105,
                              protocol['iteration'],common/'games'/f"iter_{protocol['iteration']:06d}",options)
        if games!=88:raise RuntimeError('incomplete common replay batch')
        samples=0
        for path in (common/'games').rglob('game_*.json'):
            for sample in json.loads(path.read_text())['samples']:
                for field in ('z','rootq','a0gb'):
                    value=sample[field]
                    if not math.isfinite(value) or not -1e-5 <= value <= 1+1e-5:
                        raise ValueError(f'invalid {field} target in {path}')
                samples+=1
        if samples<256:raise RuntimeError('common replay has too few training samples')
        state['bootstrap']=dict(games=games,plies=plies,samples=samples,
                                seconds=time.time()-state['bootstrap_started_at'])
        save(out/'status.json',state)
    finally:
        for proc in servers:
            if proc.poll() is None:proc.terminate()
        for proc in servers:proc.wait(timeout=15)
        for log in logs:log.close()
    return common


def check_training(branch, iteration, iterations, start_step):
    import torch
    run=Path(branch['run_dir'])
    text=(run/'loop.log').read_text()
    if not text.rstrip().endswith('[loop] finished'):
        raise RuntimeError(f"{branch['name']} did not finish cleanly")
    checkpoint=run/'ckpt'/f'iter_{iteration+iterations:04d}.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    expected=start_step+200*iterations
    if state['step']!=expected or state['config']['value_target']!=branch['name']:
        raise RuntimeError('saved step/target does not match the experiment')
    files=list((run/'games'/f'iter_{iteration+iterations:06d}').rglob('game_*.json'))
    if len(files)!=88:raise RuntimeError('final self-play iteration is incomplete')
    return str(checkpoint),expected


def supervise(out):
    from simulation_budget_experiment import running_workloads
    protocol=json.loads((out/'protocol.json').read_text())
    state=dict(status='starting',pid=os.getpid(),started_at=time.time(),branches={})
    status=out/'status.json';active=None;proc=None
    def stop(signum,frame):raise SystemExit(f'signal {signum}; finishing active iteration')
    signal.signal(signal.SIGTERM,stop)
    try:
        if running_workloads():raise RuntimeError('competing training/evaluation workload')
        save(status,state)
        common=bootstrap(protocol,out,state)
        if running_workloads():raise RuntimeError('competing workload after bootstrap')
        state['status']='preparing_branches';save(status,state)
        manifest=prepare_targets(common,protocol['iteration'],out/'branches',out/'bin/chushogi-az',protocol['iterations'])
        import torch
        initial=torch.load(common/'ckpt'/f"iter_{protocol['iteration']:04d}.pt",map_location='cpu',weights_only=True)
        start_step=int(initial['step']);del initial
        state['start_step']=start_step
        evaluation=json.loads((out/'branches/evaluation_command.json').read_text())
        evaluation += ['--gen',protocol['generator'],'--seed','20261105']
        evaluation[evaluation.index('--workers')+1]='44'
        save(out/'evaluation_command.json',evaluation)
        state.update(status='training',run_order=manifest['run_order']);save(status,state)
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CHUSHOGI_EVAL_POLICY':'legal'}
        for name in manifest['run_order']:
            if running_workloads():raise RuntimeError('competing workload before branch launch')
            branch=next(b for b in manifest['branches'] if b['name']==name)
            active=Path(branch['run_dir'])
            if (active/'STOP').exists():raise RuntimeError('external STOP request')
            info=dict(status='training',started_at=time.time())
            state['branches'][name]=info;state['active_branch']=name;save(status,state)
            with (active/'console.log').open('w') as log:
                proc=subprocess.Popen(branch['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
                info['pid']=proc.pid;save(status,state)
                code=proc.wait()
            if code or (active/'STOP').exists():raise RuntimeError(f'{name} failed or stopped; no later branches launched')
            checkpoint,step=check_training(branch,protocol['iteration'],protocol['iterations'],start_step)
            info.update(status='complete',finished_at=time.time(),checkpoint=checkpoint,step=step)
            save(status,state)
        if running_workloads():raise RuntimeError('competing workload before evaluation')
        state.pop('active_branch',None)
        state.update(status='evaluating',evaluation_started_at=time.time());save(status,state)
        with (out/'final_strength.log').open('w') as log:
            proc=subprocess.Popen(evaluation,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
            state['evaluation_pid']=proc.pid;save(status,state)
            code=proc.wait()
        state.update(status='complete' if code==0 else 'evaluation_failed',finished_at=time.time());save(status,state)
        if code:raise RuntimeError('tournament failed; inspect final_strength.log')
    except BaseException as error:
        if active is not None and proc is not None and proc.poll() is None:
            (active/'STOP').write_text('supervisor stopped; finish iteration\n')
        state.update(status='failed',error=str(error),finished_at=time.time());save(status,state)
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path)
    ap.add_argument('--iteration',type=int,default=182)
    ap.add_argument('--az',type=Path)
    ap.add_argument('--gen',type=Path,default=ROOT/'chushogi-gen')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--iterations',type=int,default=20)
    ap.add_argument('--launch',action='store_true')
    ap.add_argument('--supervise',action='store_true',help=argparse.SUPPRESS)
    a=ap.parse_args();out=a.out.resolve()
    if a.supervise:supervise(out);return
    if not a.source or not a.az or a.iteration<0 or a.iterations<=0:
        ap.error('source and az required; counts must be positive')
    source=a.source.resolve(strict=True);binary=a.az.resolve(strict=True)
    checkpoint=source/'ckpt'/f'iter_{a.iteration:04d}.pt'
    checkpoint.resolve(strict=True)
    generator=a.gen.resolve(strict=True)
    from simulation_budget_experiment import running_workloads
    if a.launch and running_workloads():raise RuntimeError('other jobs active')
    out.mkdir(parents=True,exist_ok=False)
    code=out/'source/az';code.mkdir(parents=True)
    for path in (ROOT/'az').glob('*.py'):shutil.copy2(path,code/path.name)
    (out/'bin').mkdir();shutil.copy2(binary,out/'bin/chushogi-az')
    from fork_training import sha256
    protocol=dict(source=str(source),iteration=a.iteration,iterations=a.iterations,
        python=sys.executable,generator=str(generator),checkpoint_sha256=sha256(checkpoint),
        engine_sha256=sha256(out/'bin/chushogi-az'),
        python_source_sha256={p.name:sha256(p) for p in sorted(code.glob('*.py'))},
        bootstrap_games=88,training_games_per_branch=88*a.iterations,
        target_steps_per_branch=200*a.iterations,evaluation_games=600,
        fresh_replay_reason='Old A0GB extraction stopped before entering the final leaf.')
    save(out/'protocol.json',protocol)
    if a.launch:
        with (out/'supervisor.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(code/Path(__file__).name),'--out',str(out),'--supervise'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CHUSHOGI_EVAL_POLICY':'legal'})
        save(out/'process.json',dict(pid=proc.pid))
        print(f'Started value-target experiment, PID {proc.pid}; status: {out}/status.json')
    else:print(f'Prepared frozen code and protocol: {out}')

if __name__=='__main__':main()
