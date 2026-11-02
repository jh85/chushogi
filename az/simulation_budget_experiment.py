#!/usr/bin/env python3
"""Compare self-play simulation budgets by checkpoints available after equal time.

Prepare only by default. --launch runs branches sequentially, each with the
whole machine, then a common tournament. At each deadline, a STOP file asks the
loop to finish its iteration. Any checkpoint produced after the deadline is
excluded from the primary comparison; boundary overrun is recorded explicitly.
"""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from fork_training import ROOT,prepare
from sampling_experiment import check_engine


def options_to_command(options):
    command=[options['python'],str(ROOT/'az/run_loop.py'),'--resume']
    for key,value in options.items():
        if key in ('resume','init_from'):continue
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:command.append(flag)
        else:command.extend([flag,str(value)])
    return command


def select_checkpoint(run, initial_iteration, deadline):
    eligible=[p for p in (run/'ckpt').glob('iter_*.pt')
              if int(p.stem.split('_')[1])>=initial_iteration and p.stat().st_mtime<=deadline]
    if not eligible:raise ValueError('no checkpoint available before deadline')
    return max(eligible,key=lambda p:int(p.stem.split('_')[1]))


def prepare_experiment(source,iteration,out,binary,hours=24,seed=20261030):
    cfg=json.loads((source/'loop_config.json').read_text())
    specs=[dict(name=f'sims_{n}',sims=n,root_policy_temp=1.25,temp_plies=-1,
                strength_every=0,dynamic_games=True,eval_max_batch=128,
                eval_batch_wait_ms=1.0,eval_tcp_nodelay=True) for n in (256,1024,4096)]
    manifest=prepare(source,iteration,out,binary,iterations=1000000,branch_specs=specs,
                     workers=44,games=2,eval_servers=4)
    for branch,spec in zip(manifest['branches'],specs):
        options={**cfg,**spec};options.pop('name')
        options.update(run_dir=branch['run_dir'],az=str(out.resolve()/'bin/chushogi-az'),
                       workers=44,games=2,eval_servers=4,iterations=1000000,resume=True,
                       train_lock=str(out.resolve()/'gpu-training.lock'),
                       stop_file=str(Path(branch['run_dir'])/'STOP'))
        options['python']=cfg.get('python',sys.executable)
        branch['options']=options;branch['command']=options_to_command(options)
    order=[b['name'] for b in manifest['branches']];random.Random(seed).shuffle(order)
    manifest.update(experiment='selfplay_search_budget',hours_per_branch=hours,
                    run_order=order,order_seed=seed,
                    primary_endpoint='Latest checkpoint completed no later than each equal-time deadline',
                    periodic_strength_disabled=True,
                    limitations=['One training seed; repeat before drawing broad conclusions.',
                      'Safe boundary shutdown can overrun the budget; over-budget checkpoints are not scored.',
                      'Wall time includes training and replay loading. Setup snapshot copying is excluded.',
                      'Same step-based LR schedule; branches completing more updates advance further on it.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Fixed iteration numbers would be wrong here: branches finish different counts.
    (out/'evaluation_command.json').unlink()
    return manifest


def running_workloads():
    text=subprocess.check_output(['ps','-eo','pid=,args='],text=True)
    names=['az/run_loop.py','az/checkpoint_tournament.py','az/benchmark_search.py','az/train.py','az/eval_server.py']
    return [line.strip() for line in text.splitlines()
            if any(name in line for name in names) and int(line.split()[0])!=os.getpid()]


def run_experiment(out):
    out=out.resolve();manifest=json.loads((out/'manifest.json').read_text())
    state_path=out/'experiment_status.json'
    if state_path.exists():raise ValueError('experiment already started; inspect its status before resuming')
    state=dict(status='starting',branches={})
    def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    proc=None;marker=None
    def stop_signal(signum,frame):raise SystemExit(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop_signal)
    try:
        if running_workloads():raise RuntimeError('Other training/evaluation jobs are active; allocate the machine first')
        save()
        for name in manifest['run_order']:
            if running_workloads():raise RuntimeError('Unexpected competing workload; not launching another branch')
            branch=next(b for b in manifest['branches'] if b['name']==name)
            run=Path(branch['run_dir']);marker=Path(branch['options']['stop_file'])
            if marker.exists():raise RuntimeError(f'Existing STOP request: {marker}')
            start=time.time();deadline=start+manifest['hours_per_branch']*3600
            with open(run/'console.log','a') as log:
                proc=subprocess.Popen(branch['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                      start_new_session=True,env=env)
            info=dict(pid=proc.pid,started_at=start,deadline=deadline)
            state.update(status='training',active_branch=name);state['branches'][name]=info;save()
            while proc.poll() is None and time.time()<deadline and not marker.exists():
                time.sleep(min(5,max(.01,deadline-time.time())))
            if marker.exists():raise RuntimeError('External STOP request; experiment cancelled, not launching further branches')
            if proc.poll() is not None:raise RuntimeError(f'{name} exited before deadline; inspect console.log')
            token='equal-time simulation experiment deadline\n'
            with marker.open('x') as f:f.write(token)
            state['status']='finishing_iteration';save()
            while proc.poll() is None:time.sleep(5)
            if proc.returncode or (run/'loop.log').read_text().splitlines()[-1]!='[loop] finished':
                raise RuntimeError(f'{name} failed while stopping')
            checkpoint=select_checkpoint(run,manifest['source_iteration'],deadline)
            iteration=int(checkpoint.stem.split('_')[1])
            info.update(finished_at=time.time(),boundary_overrun_seconds=max(0,time.time()-deadline),
                        selected_checkpoint=str(checkpoint),selected_iteration=iteration,
                        completed_iterations_within_budget=iteration-manifest['source_iteration'],
                        optimizer_updates_within_budget=(iteration-manifest['source_iteration'])*branch['options']['train_steps'])
            save();time.sleep(3)
        anchor=str(out/manifest['branches'][0]['name']/'ckpt/anchor.pt')
        selected=[state['branches'][b['name']]['selected_checkpoint'] for b in manifest['branches']]
        cfg=manifest['branches'][0]['options']
        evaluation=[cfg['python'],str(ROOT/'az/checkpoint_tournament.py'),'--checkpoints',anchor,*selected,
            '--pairs','50','--sims','1024','--workers','44',
            '--mate-nodes',str(cfg.get('mate_nodes',5000)),'--probe-depth',str(cfg.get('probe_depth',3)),
            '--probe-nodes',str(cfg.get('probe_nodes',2000)),
            '--eval-max-batch','128','--eval-batch-wait-ms','1','--eval-tcp-nodelay',
            '--az',str(out/'bin/chushogi-az'),'--out',str(out/'final_strength')]
        (out/'evaluation_command.json').write_text(json.dumps(evaluation,indent=2)+'\n')
        state['status']='evaluating';save()
        with open(out/'final_strength.log','a') as log:
            result=subprocess.run(evaluation,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        state.update(status='complete' if result.returncode==0 else 'evaluation_failed');save()
    except BaseException as error:
        if proc is not None and proc.poll() is None and marker is not None and not marker.exists():
            try:
                with marker.open('x') as f:f.write('experiment supervisor stopped; finish iteration\n')
            except FileExistsError:pass
        state.update(status='failed',error=str(error));save()
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path)
    ap.add_argument('--iteration',type=int,default=139)
    ap.add_argument('--az',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--hours-per-branch',type=float,default=24)
    ap.add_argument('--order-seed',type=int,default=20261030)
    ap.add_argument('--launch',action='store_true')
    ap.add_argument('--run-prepared',action='store_true',help=argparse.SUPPRESS)
    a=ap.parse_args()
    if a.run_prepared:
        run_experiment(a.out);return
    import math
    if not a.source or not a.az or a.iteration<0 or not math.isfinite(a.hours_per_branch) or a.hours_per_branch<=0:
        ap.error('source/az required; iteration nonnegative; hours finite and positive')
    check_engine(a.az.resolve(strict=True))
    manifest=prepare_experiment(a.source,a.iteration,a.out,a.az,a.hours_per_branch,a.order_seed)
    print('Prepared three sequential branches:',', '.join(manifest['run_order']))
    print(f'{a.hours_per_branch:g} hours per branch, plus safe shutdown overrun and final evaluation.')
    if a.launch:
        if running_workloads():raise RuntimeError('Other jobs active; experiment prepared but not launched')
        with open(a.out/'experiment.log','a') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--out',str(a.out.resolve()),'--run-prepared'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        (a.out/'experiment_process.json').write_text(json.dumps(dict(pid=proc.pid),indent=2)+'\n')
        print('Experiment supervisor PID:',proc.pid)

if __name__=='__main__':main()
