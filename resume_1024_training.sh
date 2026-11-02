#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CHUSHOGI_EVAL_POLICY=legal
make chushogi-az
exec /venv/main/bin/python - <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path('az').resolve()))
from fork_training import prepare, launch
from simulation_budget_experiment import running_workloads

root = Path.cwd()
out = root/'az/data/training_1024_34600_to38600'
if out.exists():
    raise SystemExit(f'Already prepared: {out}. Inspect processes.json and pilot_status.json; do not launch twice.')
if running_workloads():
    raise SystemExit('Other training/evaluation jobs are active; not launching another workload.')
manifest = prepare(root/'az/data/simulation_budget_experiment/sims_1024',172,out,
    root/'chushogi-az',iterations=20,workers=44,games=2,eval_servers=4,
    branch_specs=[dict(name='main',sims=1024,root_policy_temp=1.25,temp_plies=-1,
                       strength_every=0,dynamic_games=True,eval_max_batch=128,
                       eval_batch_wait_ms=1.0,eval_tcp_nodelay=True)])
manifest['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
run = out/'main'
# Match the anchor, midpoint and final checkpoint only after training finishes.
command = json.loads((out/'evaluation_command.json').read_text())
start = command.index('--checkpoints')+1
end = command.index('--az')
command[start:end] = [str(run/'ckpt/anchor.pt'),str(run/'ckpt/iter_0182.pt'),str(run/'ckpt/iter_0192.pt')]
command[command.index('--workers')+1] = '44'
command += ['--gen',str(root/'chushogi-gen')]
(out/'evaluation_command.json').write_text(json.dumps(command,indent=2)+'\n')
if running_workloads():
    raise SystemExit('A competing workload started during preparation; prepared run was not launched.')
processes = launch(out)
with (out/'watcher.log').open('w') as log:
    watcher = subprocess.Popen([sys.executable,str(root/'az/watch_training_fork.py'),
        '--experiment',str(out)],cwd=root,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,env=os.environ.copy())
(out/'watcher_process.json').write_text(json.dumps(dict(pid=watcher.pid),indent=2)+'\n')
print('Training:',json.dumps(processes))
print('Tournament watcher PID:',watcher.pid)
print('Run:',run)
print('Training: steps 34600 to 38600; tournament at steps 34600, 36600 and 38600.')
PY
