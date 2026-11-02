#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec /venv/main/bin/python az/tune_inference_servers.py \
  --run-dir az/data/sampling_experiment/all_moves \
  --old-pid 296480 --final-iteration 139 \
  --servers 2 3 4 --rounds 4 \
  --out az/data/throughput_audit/server_tuning_live
