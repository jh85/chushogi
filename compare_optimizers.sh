#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CHUSHOGI_EVAL_POLICY=legal
exec /venv/main/bin/python az/optimizer_experiment.py \
  --source az/data/value_target_experiment/branches/a0gb \
  --iteration 202 --iterations 20 \
  --az az/data/value_target_experiment/branches/bin/chushogi-az \
  --gen ./chushogi-gen --out az/data/optimizer_experiment --launch
