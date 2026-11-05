#!/usr/bin/env bash
# Confirm hybrid Muon/AdamW vs AdamW: 200 paired openings, 400 games.
# Usage: ./confirm_optimizers.sh [--check]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --check ) ]]; then
  echo 'Usage: ./confirm_optimizers.sh [--check]' >&2
  exit 2
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CHUSHOGI_EVAL_POLICY=legal

run='az/data/optimizer_experiment'
out="$run/confirmation_strength_seed20261107"
python='/venv/main/bin/python'
engine="$run/branches/bin/chushogi-az"
tournament="$run/source/az/checkpoint_tournament.py"
checkpoints=(
  "$run/branches/adamw/ckpt/iter_0222.pt"
  "$run/branches/muon_adamw/ckpt/iter_0222.pt"
)
for path in "${checkpoints[@]}" "$tournament"; do
  [[ -r "$path" ]] || { echo "Missing checkpoint or tournament script: $path" >&2; exit 1; }
done
for path in "$python" "$engine" './chushogi-gen'; do
  [[ -x "$path" ]] || { echo "Missing executable: $path" >&2; exit 1; }
done
if [[ -e "$out" ]]; then
  echo "Output already exists: $out. Inspect its results before launching again." >&2
  exit 1
fi
command=( "$python" "$tournament"
  --checkpoints "${checkpoints[@]}"
  --pairs 200 --seed 20261107 --opening-plies 8
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
printf 'Starting 400 games (hybrid is scored against AdamW). Log: %s.log\n' "$out"
"${command[@]}" 2>&1 | tee "$out.log"

# Resample whole color-paired openings, retaining within-pair dependence.
"$python" - "$out" <<'PY_ANALYZE'
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
out=Path(sys.argv[1])
records=[json.loads(p.read_text()) for p in out.glob('game_*.json')]
if len(records)!=400:
    raise SystemExit(f'Expected 400 games, found {len(records)}')
pairs=defaultdict(list)
for record in records:
    if Path(record['later']).parts[-3]!='muon_adamw' or Path(record['earlier']).parts[-3]!='adamw':
        raise SystemExit('Unexpected checkpoint order in tournament records')
    pairs[record['pair']].append(record)
if len(pairs)!=200 or any(len(rows)!=2 or {r['color'] for r in rows}!={0,1} for rows in pairs.values()):
    raise SystemExit('Incomplete or duplicated opening pairs')
scores=np.array([sum(r['score'] for r in rows)/2 for _,rows in sorted(pairs.items())])
rng=np.random.default_rng(20261107)
boot=np.concatenate([scores[rng.integers(0,len(scores),(5000,len(scores)))].mean(axis=1) for _ in range(10)])
report=dict(candidate='muon_adamw',opponent='adamw',games=400,opening_pairs=200,
    wins=sum(r['score']==1 for r in records),draws=sum(r['score']==.5 for r in records),
    losses=sum(r['score']==0 for r in records),score=float(scores.mean()),
    paired_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist(),
    method='Percentile bootstrap over whole opening pairs, 50000 resamples; fixed checkpoint comparison, one opening seed.')
(out/'paired_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY_ANALYZE
