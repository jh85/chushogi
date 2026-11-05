#!/usr/bin/env bash
# Five checkpoints, ten matchups, 100 paired openings each: 2,000 games.
# Usage: ./confirm_checkpoints.sh [--check]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --check ) ]]; then
  echo 'Usage: ./confirm_checkpoints.sh [--check]' >&2
  exit 2
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CHUSHOGI_EVAL_POLICY=legal

run='az/data/training_1024_34600_to38600'
out="$run/confirmation_strength_seed20261103"
python='/venv/main/bin/python'
engine="$run/bin/chushogi-az"
checkpoints=(
  "$run/main/ckpt/anchor.pt"
  "$run/main/ckpt/iter_0177.pt"
  "$run/main/ckpt/iter_0182.pt"
  "$run/main/ckpt/iter_0187.pt"
  "$run/main/ckpt/iter_0192.pt"
)
for path in "${checkpoints[@]}"; do
  [[ -r "$path" ]] || { echo "Missing checkpoint: $path" >&2; exit 1; }
done
for path in "$python" "$engine" './chushogi-gen'; do
  [[ -x "$path" ]] || { echo "Missing executable: $path" >&2; exit 1; }
done
if [[ -e "$out" ]]; then
  echo "Output already exists: $out. Inspect its results before launching again." >&2
  exit 1
fi
command=( "$python" az/checkpoint_tournament.py
  --checkpoints "${checkpoints[@]}"
  --pairs 100 --seed 20261103 --opening-plies 8
  --sims 1024 --mate-nodes 5000 --probe-depth 3 --probe-nodes 2000
  --workers 44 --eval-max-batch 128 --eval-batch-wait-ms 1 --eval-tcp-nodelay
  --az "$engine" --gen ./chushogi-gen --out "$out"
)
if [[ "${1:-}" == --check ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
"$python" - <<'PY'
import sys
sys.path.insert(0,'az')
from simulation_budget_experiment import running_workloads
jobs=running_workloads()
if jobs:
    raise SystemExit('Other training/evaluation jobs are active:\n'+'\n'.join(jobs))
PY
# Each completed game is saved immediately; the full summary is written at end.
printf 'Starting 2,000 games. Log: %s.log\n' "$out"
"${command[@]}" 2>&1 | tee "$out.log"
