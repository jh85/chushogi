#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Resume iteration 81 through 139: 16,200 -> 28,000 optimizer steps.
# Run once; --iterations counts additional iterations on each invocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

exec /venv/main/bin/python az/run_loop.py \
  --run-dir az/data/sampling_experiment/all_moves \
  --az az/data/sampling_experiment/bin/chushogi-az \
  --resume --iterations 59 \
  --workers 44 --games 2 --eval-servers 1 \
  --dynamic-games --eval-max-batch 128 \
  --eval-batch-wait-ms 1 --eval-tcp-nodelay \
  --root-policy-temp 1.25 --temp-plies -1 \
  --sims 256 --batch 16 \
  --train-steps 200 --train-batch 256 \
  --max-positions 1000000 --value-target z \
  --lr 0.001 --lr-decay-steps 200000 --seed 1 \
  --mate-nodes 5000 --probe-depth 3 --probe-nodes 2000 \
  --strength-every 10 --strength-pairs 20 \
  --strength-sims 256 --strength-workers 4 \
  --stop-file az/data/sampling_experiment/all_moves/STOP
