#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Usage: ./compare_simulation_budgets.sh [hours_per_branch]
# Run once. Default: 24 hours each, sequentially, then common evaluation.
budget_hours="${1:-24}"
output_dir="az/data/simulation_budget_experiment"
if [[ -e "$output_dir" ]]; then
  echo "Experiment directory already exists: $output_dir" >&2
  echo "Inspect its experiment_status.json before launching anything again." >&2
  exit 1
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
make AZBIN=/tmp/chushogi-search-budget /tmp/chushogi-search-budget
exec /venv/main/bin/python az/simulation_budget_experiment.py \
  --source az/data/sampling_experiment/all_moves \
  --iteration 139 \
  --az /tmp/chushogi-search-budget \
  --out "$output_dir" \
  --hours-per-branch "$budget_hours" \
  --launch
