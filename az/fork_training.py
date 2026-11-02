#!/usr/bin/env python3
"""Snapshot a completed training iteration into matched temperature branches.

Preparation never stops the source run or launches training. Use --launch only
when ready to allocate the machine to the experiment. Replay copies preserve
mtime because Buffer selects its window by timestamp.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(source, iteration, out, binary, iterations=20, branch_specs=None,
            workers=22, games=4, eval_servers=2):
    source, out, binary = source.resolve(), out.resolve(), binary.resolve(strict=True)
    checkpoint = source / 'ckpt' / f'iter_{iteration:04d}.pt'
    checkpoint.resolve(strict=True)
    cfg = json.loads((source / 'loop_config.json').read_text())
    replay_dirs = sorted(p for p in (source / 'games').glob('iter_*')
                         if p.is_dir() and int(p.name.split('_')[1]) <= iteration)
    if not replay_dirs or int(replay_dirs[-1].name.split('_')[1]) != iteration:
        raise ValueError('replay history for the requested iteration is missing')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'bin').mkdir()
    engine = out / 'bin' / 'chushogi-az'
    shutil.copy2(binary, engine)
    source_hash = sha256(checkpoint)
    manifest = dict(source=str(source), source_iteration=iteration,
                    checkpoint_sha256=source_hash, engine_sha256=sha256(engine),
                    pilot_iterations=iterations, branches=[], replay=[])
    # A content manifest verifies that both independently writable replay copies
    # begin with exactly the same records and ordering timestamps.
    for folder in replay_dirs:
        for path in sorted(folder.rglob('game_*.json')):
            manifest['replay'].append(dict(path=str(path.relative_to(source / 'games')),
                                          sha256=sha256(path), mtime_ns=path.stat().st_mtime_ns))
    specs = branch_specs or [dict(name='t1', root_policy_temp=1.0),
                             dict(name='t1p25', root_policy_temp=1.25)]
    names = [spec['name'] for spec in specs]
    if len(set(names)) != len(names) or any(not n or Path(n).name != n or n in ('.','..') for n in names):
        raise ValueError('branch names must be unique directory names')
    for spec in specs:
        tag = spec['name']
        temperature = spec['root_policy_temp']
        branch = out / tag
        (branch / 'ckpt').mkdir(parents=True)
        for name in ['current.pt', 'anchor.pt', checkpoint.name]:
            shutil.copy2(checkpoint, branch / 'ckpt' / name)
            assert sha256(branch / 'ckpt' / name) == source_hash
        for folder in replay_dirs:
            shutil.copytree(folder, branch / 'games' / folder.name, copy_function=shutil.copy2)
        for rec in manifest['replay']:
            path = branch / 'games' / rec['path']
            if sha256(path) != rec['sha256'] or path.stat().st_mtime_ns != rec['mtime_ns']:
                raise RuntimeError(f'replay snapshot mismatch: {path}')
        options = {k: v for k, v in cfg.items() if k not in
                   ('run_dir', 'init_from', 'resume', 'az', 'stop_file', 'train_lock')}
        options.update(workers=workers, games=games, eval_servers=eval_servers, iterations=iterations,
                       root_policy_temp=temperature, run_dir=str(branch), az=str(engine),
                       train_lock=str(out / 'gpu-training.lock'), stop_file=str(branch / 'STOP'))
        options.update({k:v for k,v in spec.items() if k!='name'})
        command = [cfg.get('python', sys.executable), str(ROOT / 'az/run_loop.py'), '--resume']
        for key, value in options.items():
            if isinstance(value, bool):
                if value:
                    command.append('--' + key.replace('_', '-'))
                continue
            command += ['--' + key.replace('_', '-'), str(value)]
        manifest['branches'].append(dict(name=tag, temperature=temperature,
                                         run_dir=str(branch), command=command,
                                         settings={k:v for k,v in spec.items() if k!='name'}))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    final_iteration = iteration + iterations
    evaluation = [cfg.get('python', sys.executable), str(ROOT / 'az/checkpoint_tournament.py'),
                  '--checkpoints', str(out / names[0] / 'ckpt/anchor.pt'),
                  *[str(out / tag / 'ckpt' / f'iter_{final_iteration:04d}.pt') for tag in names],
                  '--az', str(engine), '--pairs', '50', '--sims', '1024', '--workers', '4',
                  '--mate-nodes', str(cfg.get('mate_nodes',5000)),
                  '--probe-depth', str(cfg.get('probe_depth',3)),
                  '--probe-nodes', str(cfg.get('probe_nodes',2000)),
                  '--eval-max-batch', str(cfg.get('eval_max_batch',0)),
                  '--eval-batch-wait-ms', str(cfg.get('eval_batch_wait_ms',1.0)),
                  *(['--eval-tcp-nodelay'] if cfg.get('eval_tcp_nodelay') else []),
                  '--out', str(out / 'final_strength')]
    (out / 'evaluation_command.json').write_text(json.dumps(evaluation, indent=2) + '\n')
    return manifest


def launch(out):
    out = out.resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    if (out / 'processes.json').exists():
        raise ValueError('experiment already launched; resume branches explicitly')
    processes = []
    for branch in manifest['branches']:
        with open(Path(branch['run_dir']) / 'console.log', 'a') as log:
            proc = subprocess.Popen(branch['command'], cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        processes.append(dict(branch=branch['name'], pid=proc.pid))
    (out / 'processes.json').write_text(json.dumps(processes, indent=2) + '\n')
    return processes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--iteration', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--az', type=Path, required=True)
    ap.add_argument('--iterations', type=int, default=20)
    ap.add_argument('--launch', action='store_true')
    a = ap.parse_args()
    if a.iteration < 0 or a.iterations <= 0:
        ap.error('iteration must be nonnegative and iterations positive')
    manifest = prepare(a.source, a.iteration, a.out, a.az, a.iterations)
    print(f'Prepared two branches from iteration {a.iteration}; {len(manifest["replay"])} shared starting games.')
    if a.launch:
        print(json.dumps(launch(a.out), indent=2))


if __name__ == '__main__':
    main()
