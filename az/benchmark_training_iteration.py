#!/usr/bin/env python3
"""Run isolated, sequential full-iteration throughput benchmarks.

Copies one replay snapshot, then hard-links its immutable records into three
branches. Newly generated games and all checkpoints remain branch-local.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def supervise(out, source, iteration, binary):
    from fork_training import sha256
    from simulation_budget_experiment import options_to_command, running_workloads
    state = dict(status='preparing', pid=os.getpid(), started_at=time.time(), branches={})
    status_path = out/'status.json'
    write_json(status_path, state)
    active = None
    try:
        if running_workloads():
            raise RuntimeError('Other training/evaluation jobs are active')
        cfg = json.loads((source/'loop_config.json').read_text())
        checkpoint = source/'ckpt'/f'iter_{iteration:04d}.pt'
        checkpoint_hash = sha256(checkpoint)
        engine = out/'bin/chushogi-az'
        engine.parent.mkdir()
        shutil.copy2(binary, engine)
        frozen_checkpoint = out/'source_checkpoint.pt'
        shutil.copy2(checkpoint, frozen_checkpoint)
        if sha256(frozen_checkpoint) != checkpoint_hash:
            raise RuntimeError('checkpoint snapshot mismatch')
        replay = out/'replay_snapshot'
        replay.mkdir()
        folders = sorted(p for p in (source/'games').glob('iter_*')
                         if p.is_dir() and int(p.name.split('_')[1]) <= iteration)
        if not folders or int(folders[-1].name.split('_')[1]) != iteration:
            raise ValueError('requested replay iteration is missing')
        records = []
        for folder in folders:
            for path in sorted(folder.rglob('game_*.json')):
                relative = path.relative_to(source/'games')
                dest = replay/relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                # Benchmarks only read archived replay. Prevent accidental edits.
                dest.chmod(0o444)
                records.append(dict(path=str(relative), bytes=dest.stat().st_size,
                                    mtime_ns=dest.stat().st_mtime_ns, sha256=sha256(dest)))
        manifest = dict(source=str(source), source_iteration=iteration,
                        checkpoint_sha256=checkpoint_hash, engine_sha256=sha256(engine),
                        replay=records, replay_bytes=sum(r['bytes'] for r in records),
                        order=[256, 1024, 4096], branches=[],
                        limitations=['One iteration per budget, one seed.',
                            'Game lengths and positions can differ with search budget.',
                            'Historical runs use different checkpoints and are not paired speedup controls.',
                            'Setup copying excluded; branch wall time includes server startup and checkpoint archival.'])
        for sims in manifest['order']:
            run = out/f'sims_{sims}'
            (run/'ckpt').mkdir(parents=True)
            shutil.copy2(frozen_checkpoint, run/'ckpt/current.pt')
            shutil.copy2(frozen_checkpoint, run/'ckpt'/checkpoint.name)
            for rec in records:
                dest = run/'games'/rec['path']
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.link(replay/rec['path'], dest)
            options = {**cfg, 'run_dir':str(run), 'az':str(engine), 'sims':sims,
                       'workers':44, 'games':2, 'eval_servers':4, 'iterations':1,
                       'resume':True, 'init_from':'', 'strength_every':0,
                       'train_lock':str(out/'gpu-training.lock'), 'stop_file':str(run/'STOP')}
            command = options_to_command(options)
            manifest['branches'].append(dict(sims=sims, options=options, command=command))
        write_json(out/'manifest.json', manifest)
        state.update(status='running', prepared_at=time.time())
        write_json(status_path, state)
        env = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
               'CHUSHOGI_EVAL_POLICY':'legal'}
        for branch in manifest['branches']:
            if running_workloads():
                raise RuntimeError('Unexpected competing training/evaluation workload')
            sims = branch['sims']
            active = Path(branch['options']['run_dir'])
            info = dict(status='running', started_at=time.time())
            state['branches'][str(sims)] = info
            state['active_budget'] = sims
            write_json(status_path, state)
            start = time.monotonic()
            with (active/'console.log').open('w') as log:
                proc = subprocess.Popen(branch['command'], cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
                info['pid'] = proc.pid
                write_json(status_path, state)
                returncode = proc.wait()
            elapsed = time.monotonic()-start
            text = (active/'loop.log').read_text()
            if returncode or not text.rstrip().endswith('[loop] finished'):
                raise RuntimeError(f'{sims} failed; inspect {active}/console.log')
            games = re.findall(r'\[loop\] iter (\d+): (\d+) games, avg plies (\d+), self-play (\d+)s', text)
            training = re.findall(r'\[loop\] iter (\d+): trained (\d+) steps in (\d+)s', text)
            if len(games) != 1 or len(training) != 1:
                raise RuntimeError('expected exactly one completed self-play/training iteration')
            game_files = list((active/'games'/f'iter_{iteration+1:06d}').rglob('game_*.json'))
            total_plies = sum(len(json.loads(p.read_text())['moves']) for p in game_files)
            if len(game_files) != 88 or int(training[0][1]) != 200:
                raise RuntimeError('unexpected game or optimizer-step count')
            import torch
            saved = torch.load(active/'ckpt/current.pt', map_location='cpu', weights_only=True)
            initial = torch.load(frozen_checkpoint, map_location='cpu', weights_only=True)
            step = saved['step']
            if step != initial['step']+200:
                raise RuntimeError('unexpected saved optimizer step')
            del saved, initial
            selfplay, train = int(games[0][3]), int(training[0][2])
            info.update(status='complete', finished_at=time.time(), total_seconds=elapsed,
                        selfplay_seconds=selfplay, training_seconds=train,
                        phase_timing_resolution_seconds=1, games=len(game_files),
                        total_plies=total_plies, average_plies=total_plies/len(game_files),
                        plies_per_selfplay_second=total_plies/max(1,selfplay),
                        optimizer_steps=200, saved_optimizer_step=step,
                        checkpoint=str(active/'ckpt'/f'iter_{iteration+1:04d}.pt'))
            write_json(status_path, state)
            print(json.dumps(dict(sims=sims, **info)), flush=True)
        reference = state['branches']['1024']['total_seconds']
        for info in state['branches'].values():
            info['iteration_time_relative_to_1024'] = info['total_seconds']/reference
        state.update(status='complete', finished_at=time.time())
        state.pop('active_budget', None)
        write_json(status_path, state)
        write_json(out/'results.json', state)
    except BaseException as error:
        # Stop requests finish the current iteration rather than leaving servers.
        if active is not None:
            (active/'STOP').write_text('benchmark supervisor stopped\n')
        state.update(status='failed', error=str(error), finished_at=time.time())
        write_json(status_path, state)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--iteration', type=int, required=True)
    ap.add_argument('--az', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--supervise', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args()
    source, binary, out = a.source.resolve(strict=True), a.az.resolve(strict=True), a.out.resolve()
    if a.supervise:
        supervise(out, source, a.iteration, binary)
        return
    out.mkdir(parents=True, exist_ok=False)
    # Freeze all Python components, so later edits cannot change later budgets.
    code = out/'source/az'
    code.mkdir(parents=True)
    for path in (ROOT/'az').glob('*.py'):
        shutil.copy2(path, code/path.name)
    command = [sys.executable, str(code/Path(__file__).name), '--supervise',
               '--source',str(source),'--iteration',str(a.iteration),'--az',str(binary),'--out',str(out)]
    with (out/'supervisor.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=ROOT,
                                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    write_json(out/'process.json',dict(pid=proc.pid, command=command))
    print(f'Benchmark supervisor PID: {proc.pid}\nStatus: {out}/status.json')

if __name__ == '__main__':
    main()
