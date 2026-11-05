#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CHUSHOGI_EVAL_POLICY=legal
make AZBIN=/tmp/chushogi-value-target /tmp/chushogi-value-target
exec /venv/main/bin/python az/value_target_experiment.py \
  --source az/data/training_1024_34600_to38600/main \
  --iteration 182 --iterations 20 \
  --az /tmp/chushogi-value-target --gen ./chushogi-gen \
  --out az/data/value_target_experiment --launch
