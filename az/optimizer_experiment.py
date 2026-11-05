#!/usr/bin/env python3
"""Frozen, sequential AdamW versus native Muon/AdamW continuation experiment."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from fork_training import prepare, sha256
from value_target_experiment import save
from simulation_budget_experiment import running_workloads

ROOT=Path(__file__).resolve().parent.parent


def choose_scale(results):
    stable={r['scale']:r['val_loss'] for r in results
            if r['optimizer']=='muon_adamw' and r['status']=='complete'
            and math.isfinite(r['val_loss'])}
    baseline=next(r['val_loss'] for r in results if r['optimizer']=='adamw' and r['status']=='complete')
    if not stable or min(stable.values())>1.2*baseline:
        raise RuntimeError('Muon stability/validation gate failed; full training not launched')
    best=min(stable,key=lambda scale:(stable[scale],abs(scale-1)))
    # Prefer the paper-matched default unless a different scale improves >1%.
    if 1. in stable and stable[1.]<=1.01*stable[best]:return 1.
    return best


def prepare_branches(source, iteration, out, binary, iterations, scale):
    shared=dict(value_target='a0gb',sims=1024,root_policy_temp=1.25,temp_plies=-1,
                strength_every=0,dynamic_games=True,eval_max_batch=128,
                eval_batch_wait_ms=1.,eval_tcp_nodelay=True,muon_warmup_steps=50)
    specs=[dict(name='adamw',optimizer='adamw',muon_lr_scale=1.,**shared),
           dict(name='muon_adamw',optimizer='muon_adamw',muon_lr_scale=scale,**shared)]
    m=prepare(source,iteration,out,binary,iterations,specs,44,2,4)
    order=[b['name'] for b in m['branches']];random.Random(20261106).shuffle(order)
    m.update(experiment='optimizer_switch',run_order=order,
        limitations=['One training seed; sequential equal updates/game counts, not equal wall time.',
          'Continuation from an AdamW-trained checkpoint, not a from-scratch optimizer comparison.',
          'Muon initializes matrix momentum; AdamW auxiliary moments are preserved.',
          'Hybrid LR scale selected using a short fixed-replay validation pilot.'])
    save(out/'manifest.json',m)
    return m


def pilot(out, protocol, state):
    folder=out/'pilot';folder.mkdir()
    source=Path(protocol['source'])
    games=sorted((source/'games'/f"iter_{protocol['iteration']:06d}").rglob('game_*.json'))
    if len(games)!=88:raise RuntimeError('expected 88 source games for pilot split')
    random.Random(20261106).shuffle(games)
    records=[]
    for i,path in enumerate(games):
        split='val' if i<8 else 'train'
        dest=folder/split/f'game_{i:03d}.json';dest.parent.mkdir(exist_ok=True)
        shutil.copy2(path,dest)
        records.append(dict(source=str(path),split=split,sha256=sha256(dest)))
    save(folder/'data_manifest.json',records)
    checkpoint=source/'ckpt'/f"iter_{protocol['iteration']:04d}.pt"
    results=[]
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for kind,scale in [('adamw',1.),('muon_adamw',.3),('muon_adamw',1.),('muon_adamw',3.)]:
        tag=f'{kind}_{scale:g}'
        state.update(status='lr_pilot',active_pilot=tag);save(out/'status.json',state)
        cmd=[protocol['python'],str(ROOT/'az/train.py'),'--games-dirs',str(folder/'train'),
            '--val-games-dir',str(folder/'val'),'--out',str(folder/f'{tag}.pt'),
            '--init-from',str(checkpoint),'--steps','100','--batch','256',
            '--max-positions','8192','--value-target','a0gb','--seed','20261106',
            '--lr','0.001','--lr-decay-steps','200000','--optimizer',kind,
            '--muon-lr-scale',str(scale),'--muon-warmup-steps','50','--log-every','10']
        started=time.time()
        with (folder/f'{tag}.log').open('w') as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        text=(folder/f'{tag}.log').read_text()
        match=re.search(r'\[train\] val loss ([0-9.eE+-]+)',text)
        item=dict(optimizer=kind,scale=scale,seconds=time.time()-started,
                  status='complete' if result.returncode==0 and match else 'failed')
        if match:item['val_loss']=float(match[1])
        results.append(item);save(folder/'results.json',results)
        if kind=='adamw' and item['status']!='complete':
            raise RuntimeError('baseline pilot failed; inspect pilot log')
    chosen=choose_scale(results)
    save(folder/'selection.json',dict(muon_lr_scale=chosen,
        policy='Choose lowest finite validation loss; prefer scale 1 if within 1%; reject >20% degradation vs baseline.',
        notes='Fixed game-disjoint pilot train/validation split. These are existing replay games, not unseen self-play. Pilot weights are discarded.'))
    state.pop('active_pilot',None)
    state['selected_muon_lr_scale']=chosen
    return chosen


def supervise(out):
    p=json.loads((out/'protocol.json').read_text())
    state=dict(status='starting',pid=os.getpid(),started_at=time.time(),branches={})
    status=out/'status.json';active=None;proc=None
    def stop(signum,frame):raise SystemExit(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    try:
        if running_workloads():raise RuntimeError('competing workload')
        save(status,state)
        scale=pilot(out,p,state)
        if running_workloads():raise RuntimeError('competing workload after pilot')
        state['status']='preparing_branches';save(status,state)
        source=Path(p['source'])
        if sha256(source/'ckpt'/f"iter_{p['iteration']:04d}.pt")!=p['checkpoint_sha256']:
            raise RuntimeError('source checkpoint changed')
        m=prepare_branches(source,p['iteration'],out/'branches',out/'bin/chushogi-az',p['iterations'],scale)
        evaluation=json.loads((out/'branches/evaluation_command.json').read_text())
        evaluation[evaluation.index('--workers')+1]='44'
        evaluation+=['--gen',p['generator'],'--seed','20261106']
        save(out/'evaluation_command.json',evaluation)
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CHUSHOGI_EVAL_POLICY':'legal'}
        for name in m['run_order']:
            if running_workloads():raise RuntimeError('competing workload before branch')
            branch=next(b for b in m['branches'] if b['name']==name)
            active=Path(branch['run_dir'])
            info=dict(status='training',started_at=time.time())
            state.update(status='training',active_branch=name,run_order=m['run_order'])
            state['branches'][name]=info;save(status,state)
            with (active/'console.log').open('w') as log:
                proc=subprocess.Popen(branch['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
                info['pid']=proc.pid;save(status,state)
                code=proc.wait()
            if code or (active/'STOP').exists():raise RuntimeError(f'{name} failed/stopped')
            import torch
            final=active/'ckpt'/f"iter_{p['iteration']+p['iterations']:04d}.pt"
            checkpoint=torch.load(final,map_location='cpu',weights_only=True)
            expected=p['start_step']+200*p['iterations']
            if checkpoint['step']!=expected or checkpoint['config']['optimizer']!=name:
                raise RuntimeError('saved optimizer/step mismatch')
            if not (active/'loop.log').read_text().rstrip().endswith('[loop] finished'):
                raise RuntimeError('training did not finish cleanly')
            for iteration in range(p['iteration']+1,p['iteration']+p['iterations']+1):
                if len(list((active/'games'/f'iter_{iteration:06d}').rglob('game_*.json')))!=88:
                    raise RuntimeError('missing training games')
            info.update(status='complete',finished_at=time.time(),step=expected,checkpoint=str(final))
            del checkpoint
            save(status,state)
        if running_workloads():raise RuntimeError('competing workload before tournament')
        active=None;state.pop('active_branch',None)
        state.update(status='evaluating',evaluation_started_at=time.time());save(status,state)
        with (out/'final_strength.log').open('w') as log:
            proc=subprocess.Popen(evaluation,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
            state['evaluation_pid']=proc.pid;save(status,state)
            code=proc.wait()
        if code:raise RuntimeError('tournament failed')
        state.update(status='complete',finished_at=time.time());save(status,state)
    except BaseException as error:
        if proc is not None and proc.poll() is None:
            if active is not None:(active/'STOP').write_text('supervisor stopped; finish iteration\n')
            else:proc.terminate()
        state.update(status='failed',error=str(error),finished_at=time.time());save(status,state)
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path)
    ap.add_argument('--iteration',type=int,default=202)
    ap.add_argument('--iterations',type=int,default=20)
    ap.add_argument('--az',type=Path)
    ap.add_argument('--gen',type=Path,default=ROOT/'chushogi-gen')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--launch',action='store_true')
    ap.add_argument('--supervise',action='store_true',help=argparse.SUPPRESS)
    a=ap.parse_args();out=a.out.resolve()
    if a.supervise:supervise(out);return
    if not a.source or not a.az or a.iteration<0 or a.iterations<=0:ap.error('source/az and positive counts required')
    import torch
    if not hasattr(torch.optim,'Muon'):raise RuntimeError('native torch.optim.Muon unavailable')
    if running_workloads():raise RuntimeError('other jobs active')
    source=a.source.resolve(strict=True);binary=a.az.resolve(strict=True)
    cp=source/'ckpt'/f'iter_{a.iteration:04d}.pt'
    initial=torch.load(cp,map_location='cpu',weights_only=True)
    start_step=int(initial['step']);del initial
    generator=a.gen.resolve(strict=True)
    out.mkdir(parents=True,exist_ok=False)
    code=out/'source/az';code.mkdir(parents=True)
    for path in (ROOT/'az').glob('*.py'):shutil.copy2(path,code/path.name)
    (out/'bin').mkdir();shutil.copy2(binary,out/'bin/chushogi-az')
    save(out/'protocol.json',dict(source=str(source),iteration=a.iteration,iterations=a.iterations,
        start_step=start_step,python=sys.executable,generator=str(generator),
        checkpoint_sha256=sha256(cp),engine_sha256=sha256(out/'bin/chushogi-az'),
        pytorch_version=str(torch.__version__),
        python_source_sha256={path.name:sha256(path) for path in sorted(code.glob('*.py'))},
        optimizer_hybrid='Muon trunk 2D matrices; AdamW all remaining parameters',
        muon_adjust_lr_fn='match_rms_adamw',muon_warmup_steps=50,
        pilot_steps=100,pilot_scales=[.3,1.,3.],training_updates_per_branch=200*a.iterations,
        evaluation_games=300))
    if a.launch:
        with (out/'supervisor.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(code/Path(__file__).name),'--out',str(out),'--supervise'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CHUSHOGI_EVAL_POLICY':'legal'})
        save(out/'process.json',dict(pid=proc.pid))
        print(f'Optimizer experiment PID {proc.pid}; status: {out}/status.json')
    else:print('Prepared frozen experiment:',out)

if __name__=='__main__':main()
