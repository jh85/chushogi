# Chu Shogi legal move generator (C++17)

A legal move generator for chu shogi (中将棋, 12x12) using the rules and
SFEN/USI conventions of lishogi.org (the source of the game records used for
testing). Behavior was implemented to match lishogi's scalashogi engine move
for move, including the lion-trading rules and the chu-shogi-specific third
SFEN field.

## Pipeline

As requested, the program is built around an explicit conversion layer:

```
initial SFEN + ordered USI moves
   -- encode (src/sfen.cpp, src/usi.cpp) -->
internal Position (src/position.h: board[144], side to move,
                   last-lion-capture square, move number)
   -- legal move generator (src/movegen.cpp) -->
internal Move list (from / mid / to / promote)
   -- decode (src/usi.cpp) -->
USI move strings
```

## Build

    make            # produces ./chushogi-gen

## Usage

Single shot:

    ./chushogi-gen "lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1" 8i8h 6c5e

prints all legal moves in USI format after the given moves.

Batch protocol on stdin (one command per line):

    position sfen <board> <b|w> <lioncap|-> <num> [moves m1 m2 ...]
    legal    -> "legal <usi moves...>"
    sfen     -> "sfen <current sfen>"
    status   -> "status <playing|royalslost|bareking|stalemate|draw> [sente|gote]"
    seed <n> -> seeds the RNG used by "go random"
    go random-> "bestmove <usi>" (uniformly random legal move; "resign" if none)
    mate [nodes] [bns|pndn]
             -> "mate <yes|no|unknown>", "nodes <n>", and on yes
                "pv <usi moves...>" (a validated forced mate line)
    move <usi>
    rules <lishogi|jcsa>
             -> "ok"; rule set for the current position and the following
                commands (default lishogi; see "JCSA rules" below)
    quit

Every move fed through `position ... moves` / `move` is validated against the
generated legal move list; an illegal move is a fatal error. Position commands
that extend the previous one apply only the new moves, so re-sending the start
position plus the full move list every turn (as in a USI match) stays cheap.

## Checkmate solver (BNS)

`src/mate.{h,cpp}` implements a checkmate checker with the **BNS algorithm**
(Branch Number Search, after Okabe Fumihiro's route-branch-number paper),
ported from the reference implementation in `/data2/cs/JHBR3/mate/bns.{h,cc}`;
a proof/disproof-number arithmetic mode (df-pn style) shares the same engine
(reference: YaneuraOu's `source/mate/mate_dfpn.hpp`) and is selectable with
`mate <nodes> pndn`.

Question answered (tsume semantics): can the side to move force the capture
of the enemy king, playing only checking moves while the defender plays only
saving moves? Because chu shogi has no check-evasion rule, a single-royal
defender who fails to evade loses on the spot, so restricting the defender
to evasions is sound. A proved "yes" is a real forced win; "no" means no
such restricted proof exists within the node limit. The search uses a
ply-keyed transposition table, treats route cycles as no-mate, and keeps
depth-cap / multi-royal results out of the table so they surface as
"unknown". Positions where the defender has several royals (king + prince)
are outside the model and report "unknown". The attacker may be kingless
(tsume problems conventionally omit the attacking king); the
capture-the-attacker escape then never applies. Every returned PV is
self-validated move by move inside the engine.

BNS vs df-pn on the tsume collection (tools/mate_compare.py, 209 provable
problems, identical node caps): verdicts never contradict; on the 178
positions both prove, df-pn spends 1.16x BNS's nodes; on the 21 disproofs
df-pn is faster (0.76x). But df-pn fails to prove 9 mates and 1 disproof
within 2x the node count BNS needs (they come back "unknown") — BNS's
branch-number thresholds are markedly better at driving deep proofs to
completion, which is why it is the default.


The branch-number core is the reference's verbatim arithmetic:
unresolved leaf `abn=obn=1`, proved mate `{0, INF}`, proved no-mate
`{INF, 0}`, OR node `abn = min abn(child)`, `obn = best.obn + (k - 1)`,
AND node dual, child thresholds `min(second + 1, ABN)` /
`OBN - (node.obn - best.obn)`.

## AlphaZero self-play agent (az/)

A from-scratch AlphaZero loop lives in `az/` (Python) + `src/az/` (C++):

- **Encoding** (`src/az/encode.*`, `az/encode.py`): positions canonicalized to
  the side-to-move's view (`Position::flipped` — mirror + color swap when gote
  is to move), 82 planes of 12x12 (39 roles x 2 sides, lion-capture square,
  repetition counts, ply-cap progress). Policy space is 50,688: the from x to
  matrix (144 x 144 = 20,736), a second from x to matrix for promotions, and
  a from x midDir x toDir block (144 x 8 x 8 = 9,216) for two-step lion-power
  moves (lion / horned falcon / soaring eagle, incl. igui). Legal-move masking
  and softmax happen outside the net.
- **MCTS** (`src/az/mcts.*`): dlshogi/JHBR3-style PUCT (c_init 1.25, c_base
  19652, FPU, virtual loss, batched leaf eval, root Dirichlet noise 0.25/0.15).
  Terminal values from the engine's status rules + in-tree repetition cutoff +
  1000-ply cap. Each root search reports the visit policy plus three value
  targets: z (game result), rootQ (soft-Z), and the A0GB greedy-path leaf.
- **Bridge**: `chushogi-az` self-play workers send bit-packed positions over
  TCP to `az/eval_server.py` (PyTorch, GPU, checkpoint --watch reload).
- **Net** (`az/model.py`): transformer over the 144 squares; policy head is
  QK^T per square (the from x to matrix) x2 + a per-square double-move MLP;
  WDL value head.
- **Training** (`az/train.py`, `az/buffer.py`): soft cross-entropy on visit
  counts + WDL CE; value target switch `z | softz | a0gb | blend(lambda)`
  (default blend 0.5). Atomic checkpoint publishing.
- **Loop**: `az/run_loop.py` — W workers x G games per iteration, then train
  and publish. `az/eval_match.py` — matches vs the random player (`go random`)
  or another checkpoint for strength tracking.

Tests: `make az-test` runs az_encode_test (C++/Python encoding agreement +
move round-trip over real data), az_selfplay_test (record validity), az_mcts_test
(mate-in-1 must be found — guards the backup-perspective convention), and
az_probe_test (mate probes: the root override plays a mate-in-1 without
search, and an in-tree probe makes MCTS avoid a move that loses the king).
az_pipe_test (untrained net end-to-end through az/eval_server.py) is not
part of it and is run separately.

``` 
make chushogi-az
python3 az/run_loop.py --workers 8 --games 2 --sims 64 --iterations 300
```

Replay uses the newest positions by file modification time, independent of
PID-based filenames. New runs store games in iteration/worker directories.
Checkpoints include AdamW state so training resumes with its moments intact;
legacy model-only checkpoints still load. Full training logs are retained per
iteration.

For a clean baseline after the search/replay fixes, start a separate run with
random weights and fresh self-play data:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 az/run_loop.py \
  --run-dir az/data/run3 \
  --workers 44 --games 2 --eval-servers 4 --sims 256 --batch 16 \
  --train-steps 200 --train-batch 256 --max-positions 1000000 \
  --lr 1e-3 --lr-decay-steps 200000 --value-target z --iterations 1000 \
  --strength-every 10 --strength-pairs 20 --strength-sims 256
```

These are starting settings, not a proven optimum: fewer updates per new
position, with the trainer's default warmup and a cosine learning-rate decay.
An optional warm-start experiment can use `--init-from CHECKPOINT` in a new
run directory; choose its seed with matches rather than loss alone.
Keep old runs separate because their search targets were generated
before neural leaf backup and deeper node expansion were fixed. Periodic
strength matches compare the candidate against the initial anchor and an older
checkpoint, with identical random openings played from both colors. Results
and move traces go under `RUN/strength/`; scores include half a point per draw.
Use sufficiently many opening pairs before concluding that strength improved.

Standalone checkpoint evaluation (all pairs, no self-play noise or mate probes):

```sh
python3 az/checkpoint_tournament.py \
  --checkpoints az/data/run2/ckpt/iter_0100.pt az/data/run2/ckpt/iter_0522.pt \
  --pairs 20 --sims 256 --workers 4 --out az/data/checkpoint_comparison
```

The output directory must be new. Each checkpoint has a fixed eval server;
the referee validates moves and adjudicates terminal positions, repetition,
and the 1000-ply limit. The same search binary is used for both sides.

To inspect repetition decisions without replacing the engine used by a live
training run, build a separate diagnostic binary:

```sh
make AZBIN=/tmp/chushogi-repetition-audit /tmp/chushogi-repetition-audit
python3 az/repetition_audit.py \
  --games-dir az/data/run3/strength/iter_000009 \
  --candidate az/data/run3/ckpt/iter_0009.pt \
  --opponent az/data/run3/ckpt/iter_0000.pt \
  --az /tmp/chushogi-repetition-audit --games 10 --sims 256 \
  --out az/data/run3/repetition_audit
```

This preserves complete move histories and compares the legacy early
repetition shortcut with fourfold counting at the same search budget. The
`analyze az --repetition early|fourfold` engine command emits root WDL,
move priors, visits, values, and repetition-cutoff visits. Production search
keeps its original early-cutoff default. Changed moves alone do not establish
better play; counterfactual continuations require separate evaluation.
For a larger-search comparison on the candidate's same saved positions, use
`--candidate-only --budgets 1024 4096` with a new `--out` directory. Keep
the earlier 256-simulation audit as the baseline; each budget starts a fresh
search using the same checkpoint, history, seed, and batch size.


To compare root policy temperatures at equal budgets without changing an active
training run:

```sh
make AZBIN=/tmp/chushogi-temperature-audit /tmp/chushogi-temperature-audit
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /venv/main/bin/python az/repetition_audit.py \
  --games-dir az/data/run3/strength/iter_000009 \
  --candidate az/data/run3/ckpt/iter_0009.pt \
  --opponent az/data/run3/ckpt/iter_0000.pt \
  --az /tmp/chushogi-temperature-audit --games 10 --candidate-only \
  --budgets 256 1024 --temperatures 1.0 1.25 --workers 2 \
  --out az/data/run3/root_temperature_audit_000009
```

Each condition starts from the same full history and random seed with root noise
and move sampling disabled. `--root-policy-temp T` applies `softmax(logits/T)`
only at the search root, before any root noise; its default is 1.0. It differs
from sampling moves from the final visit distribution. `summary.json` reports
paired move changes, choices that revisit earlier positions, exploration counts,
and visits allocated to nonrepeating children. Revisiting a position does not
necessarily mean an immediate terminal draw. These diagnostics measure search
behavior; establishing training or playing-strength benefits needs a subsequent
controlled training/match experiment. The production training CLI is unchanged.


### Controlled training forks

`az/fork_training.py` prepares independent, matched replay/checkpoint copies:

```sh
make AZBIN=/tmp/chushogi-temperature-training /tmp/chushogi-temperature-training
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /venv/main/bin/python az/fork_training.py \
  --source az/data/run3 --iteration 19 \
  --az /tmp/chushogi-temperature-training \
  --out az/data/run3_temperature_fork --iterations 20
```

The output directory must be new. Preparation preserves checkpoint optimizer
state and training step, replay contents and timestamps, and iteration numbering.
It excludes games after the selected completed iteration. `manifest.json` records
checkpoint/engine/replay hashes and exact commands. Both branches use 22 workers,
4 games per worker, 2 evaluation servers, identical seeds and other settings;
only root temperature differs. The engine is copied into the experiment directory.
Use `--launch` to also start both branches, after stopping the original run to
avoid oversubscribing the machine. Preparation alone never stops the source.
`--init-from` alone does not preserve replay history.

The running experiment is `az/data/run3_temperature_fork`: branches `t1` and
`t1p25`, each doing 20 more iterations (20 through 39) from checkpoint 19/step
4000. GPU trainer processes share a file lock; self-play runs concurrently.
See each branch's `console.log`, `loop.log`, and `loop_config.json`.
`processes.json` records their PIDs. Allow roughly 2–4 days for the pilot;
first-iteration timings will provide a better estimate.

To stop both branches cleanly after their current iteration:

```sh
touch az/data/run3_temperature_fork/t1/STOP
touch az/data/run3_temperature_fork/t1p25/STOP
```

Remove the corresponding STOP file before resuming a branch with its saved
command. `--iterations` counts additional iterations on each invocation.
The detached `az/watch_training_fork.py` watcher is already running for this
experiment. It waits for both loops to exit, summarizes only new pilot games,
and evaluates both final checkpoints and the shared anchor at 1024 simulations,
50 paired openings per matchup, with the same default evaluation temperature
1.0 and no root noise. See `pilot_status.json`, `selfplay_summary.json`, and
`final_strength/summary.json`. If a branch stops before checkpoint 39, it records
`training_incomplete` and does not launch final evaluation. Evaluation commands
are saved in `evaluation_command.json`; diagnostics alone do not establish
playing-strength gains. One pair of training branches is an exploratory pilot,
not a repeated-seed study.


### Self-play throughput

The inference server supports combining requests from different games with
`--max-batch 128 --batch-wait-ms 1 --tcp-nodelay`. This is independent of each
search's `--batch 16`: it changes GPU scheduling, not the simulation budget.
The default remains unbatched for compatibility. The server logs cumulative
request, position, batch, and largest-batch counts every 30 seconds when enabled.
GPU batches can change BF16 rounding and occasionally search choices.

The corresponding loop options are `--eval-max-batch 128
--eval-batch-wait-ms 1 --eval-tcp-nodelay --eval-servers 1`. Add
`--dynamic-games` to distribute individual games among available workers instead
of assigning each worker a fixed bundle. With `--workers 44 --games 2`, this still
generates 88 games before each training update. Original per-game seeds are
preserved: `seed + iteration*100003 + original_worker*101 + game_index + 1`.
Each game has a separate replay subdirectory. A failed or missing game aborts
self-play rather than training on an incomplete iteration.

`az/benchmark_inference.py` measures request throughput and latency with an
isolated server and frozen checkpoint. `az/benchmark_search.py` measures searches
using complete saved histories. Both require a new output directory. The
October 25 measurements are saved under `az/data/throughput_audit`: batching
improved the concurrent synthetic workloads by about 3–9 times and the 20-position
search workload by about 2.7 times. These are benchmark results, not a measured
full-iteration speedup. The old server source is preserved in that audit directory.

For the current continuation, `az/resume_at_boundary.py` has been scheduled to
hand over after the current iteration. It uses the loop's STOP file, waits for a
clean exit and matching archived/current checkpoints, then resumes with the
optimized engine and settings. It never kills running games or training updates.
The end target stays iteration 139 (28,000 optimizer steps). Status is in
`t1p25/throughput_handover.json`; new logs go to `t1p25/throughput_console.log`.
The previous loop configuration and process record are archived at the boundary.


To compare inference server counts, `az/benchmark_search.py` accepts `--servers`
and `--repeats`. Keep the checkpoint, positions, total `--workers`, and
`--max-batch` fixed; only change server count. Clients remain connected for each
trial, warmup is excluded, and jobs are distributed across servers. Trials are
saved individually with checkpoint and input hashes. The live training run is
never reconfigured by this benchmark. Use alternating/reversed condition order
to check for load-related timing variation.

### Inference output transfer and profiling

Newly built clients request only legal-move policy logits. They send canonical
policy indices; the server gathers those entries on the GPU before the float16
copy. WDL, legal move probabilities, search settings and training targets retain
their previous semantics. Legal move lists are reused for MCTS expansion.
The server also accepts the original full-policy protocol. Set
`CHUSHOGI_EVAL_POLICY=full` to force legacy requests (for comparison or an older
server); the default is `legal`. Rebuild with `make chushogi-az` and restart
inference servers to use both updated components.

At 1024 simulations, 44 workers, four servers and batch limit 128, three trials
on the same 88 saved histories gave median times of 5.90 seconds for full policy
and 3.89 seconds for legal logits: 1.52x search throughput, or 34% less time.
Six single-client searches produced exactly identical search outputs across
protocols, excluding timing fields. CPU/CUDA output tests verify byte equality
of requested logits. Full training-iteration improvement remains unmeasured.
Results: `az/data/pipeline_profile_1024/legal_policy_summary.json`; use
`az/benchmark_search.py --eval-policy full|legal` for controlled comparisons.


Inference outputs are converted to float16 on the GPU before copying to CPU.
This preserves the fp16 wire format while removing the previous float32 copy
and CPU conversion. CPU/CUDA tests compare output bytes, including signed zero,
rounding, underflow, and overflow. No search or learning setting changes.

A November 2 fixed-position profile at 1024 simulations, 44 workers and four
servers found about 88% of search wall time inside evaluator calls, with mean
GPU utilization around 36%. This includes batching waits, client/server work and
transfers; it is not all GPU arithmetic. The output-conversion change improved
median search time from 9.11 to 5.87 seconds on 88 representative positions over
repeated trials (about 1.55x throughput). Full-iteration improvement is unmeasured.
Raw results and telemetry are in `az/data/pipeline_profile_1024`.

`analyze az` emits `search_wall_seconds`, `search_cpu_seconds`,
`evaluator_wall_seconds`, `evaluator_cpu_seconds`, evaluator calls and evaluated
positions. `az/benchmark_search.py --profile-eval` also enables server phase
measurements. Profiling is diagnostic and adds overhead; normal self-play and
server operation keep it disabled. GPU stream timing includes scheduling gaps
and must not be interpreted as pure kernel arithmetic time.

### Mate searches in checkpoint evaluation

Checkpoint tournaments now default to the self-play tactical settings:
`--mate-nodes 5000 --probe-depth 3 --probe-nodes 2000`. Both players use the
same budgets, while final move selection remains greedy, root policy temperature
remains 1.0, and root noise stays disabled. Periodic evaluations from
`az/run_loop.py` explicitly inherit that run's mate budgets and inference-batching
settings. The standalone engine's `go az` defaults remain mate-disabled.

For the historical mate-disabled evaluation, pass `--mate-nodes 0 --probe-depth 0`.
Tournament config files record these choices; preserve old result directories
rather than overwriting or pooling results from different tactical configurations.
`az/benchmark_search.py` accepts the same mate settings plus `--sims` and
`--root-policy-temp` to measure per-search cost. Actual tournament runtime also
depends on how the settings change game lengths.

### Selecting the inference server count

`tune_and_resume_training.sh` runs `az/tune_inference_servers.py` for the current
continuation. Run it as the training directory's owner (currently root), ideally
inside tmux. It writes a STOP marker, waits for a clean iteration boundary,
verifies the archived checkpoint, then benchmarks 2, 3, and 4 batching servers.
The same frozen checkpoint, 88 sampled game histories, 44 search workers, and
128-position batching limit are used for every condition. Four rounds alternate
ascending and descending server counts. It selects the lowest median search time
and resumes with that server count, preserving the other learning settings and
the final target iteration 139/optimizer step 28000. Results and status go to
`az/data/throughput_audit/server_tuning_live`; resumed output goes to
`az/data/sampling_experiment/all_moves/server_tuned_console.log`.

The launcher is configured for the current PID 296480. An invalid/stale PID or a
pre-existing STOP marker causes it to refuse the operation. It does not kill jobs.
Failures are recorded and leave the run paused for inspection. A successful
benchmark measures search throughput; compare subsequent iteration timings to
measure the actual training-loop improvement. The current session cannot launch
this handover because its account cannot write the root-owned run directory.

### Comparing self-play search budgets under equal time

Run `./compare_simulation_budgets.sh` inside tmux to prepare and launch sequential
256-, 1024-, and 4096-simulation branches from iteration 139/optimizer step 28000.
The default is 24 hours per branch. For example,
`./compare_simulation_budgets.sh 12` sets 12 hours per branch. Snapshot preparation
runs in the foreground; the experiment supervisor is detached after preparation.
The output directory `az/data/simulation_budget_experiment` must be new.

Each branch gets the full machine: 44 self-play workers and 4 batching inference
servers. All inherit the same initial checkpoint/optimizer and independent replay
copies with preserved timestamps. All generate 88 games before 200 optimizer
updates, keep root temperature 1.25 and full-game sampling, and use identical mate
budgets and the same step-based learning-rate schedule. Simulation count is the
only learning-setting difference. Periodic tournaments are disabled during the
timed experiment so they do not consume different shares of each branch's budget.
The seeded branch order is recorded in the manifest. A competing known training
or evaluation job causes launch to refuse rather than oversubscribing the machine.

The comparison uses checkpoints **available by each branch's deadline**. At that
deadline the supervisor creates a STOP marker and allows the current iteration to
finish safely. Any checkpoint completed afterward is excluded from primary
scoring, and the shutdown overrun is recorded. Therefore total experiment duration
is three budgets plus boundary overruns and final evaluation. A budget too short
for one iteration can leave a branch's scored checkpoint at the original anchor.
The script never interrupts or modifies the source run.

The supervisor records actual optimizer updates completed within each budget in
`experiment_status.json`; it does not assume exact 4x or 16x timing ratios. A
common 1024-simulation, greedy, mate-enabled tournament compares the three selected
checkpoints and the shared starting agent using 50 paired openings per matchup.
See `final_strength/summary.json` and `final_strength.log`. This is a single-seed
pilot. GPU-hours exclude initial replay-copy preparation, include replay loading
and optimizer training, and exclude the reported safe-shutdown overrun.

The underlying `az/simulation_budget_experiment.py` prepares only by default;
`--launch` starts the sequential supervisor. `--source`, `--iteration`, `--az`,
`--out`, `--hours-per-branch`, and `--order-seed` configure a fresh experiment.
Existing experiment directories are not automatically restarted. On a supervisor
failure, inspect its status before doing anything else; it requests a boundary
stop for the active branch and does not start subsequent branches.

### Opening-only versus full-game sampling experiment

`--temp-plies 20` retains the existing behavior: sample moves proportional to
MCTS root visits for the first 20 plies, then choose the most-visited move.
`--temp-plies -1` samples from that visit distribution throughout the game;
`--temp-plies 0` always selects the most-visited move. This is separate from
`--root-policy-temp 1.25`, which softens root network priors before search.
The saved policy training target remains the normalized visit distribution.
Game JSON records both settings under `selfplay` for auditing.

Build a separate engine and prepare a matched experiment from a completed
checkpoint, without changing the running training job:

```sh
make AZBIN=/tmp/chushogi-sampling /tmp/chushogi-sampling
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /venv/main/bin/python az/sampling_experiment.py \
  --source az/data/run3_temperature_fork/t1p25 --iteration 39 \
  --az /tmp/chushogi-sampling --out az/data/sampling_experiment \
  --iterations 20
```

The example uses the immutable checkpoint at iteration 39/optimizer step 8000.
Another completed iteration can be selected explicitly. It creates independent
`opening20` and `all_moves` branches from identical checkpoint/optimizer and
replay snapshots, preserving replay timestamps and numbering. Both branches use
root policy temperature 1.25, 22 workers, 4 games per worker (88 games per
iteration), one inference server, and the same inherited inference/training
settings. The only learning-setting difference is the move sampling cutoff.
The engine must support the new flag; an older binary is rejected before copying
data. The output directory must be new. This is a fresh fork; use the saved
branch commands to resume an interrupted experiment rather than preparing again.

Add `--launch` to the preparation command to also launch both loops and an
automatic final-evaluation watcher. Preparation alone never starts training.
Running these two branches alongside the current 44-worker continuation would
compete for the machine; launch when resources have been allocated for the pilot.
GPU trainer updates share a lock between the two experiment branches.
`--workers` and `--games` override per-branch allocation, equally for both branches.

For the example, each branch runs iterations 40–59, reaching 12000 optimizer
steps. Final evaluation compares both checkpoints and their shared starting
checkpoint at 1024 simulations, with 50 paired openings per matchup, greedy
move selection, root temperature 1.0, and no root noise. Results are written to
`selfplay_summary.json` and `final_strength/summary.json`; `pilot_status.json`
tracks completion. Evaluate playing strength and ability to finish games, not
only draw rate or training loss. A single pair of training runs is an exploratory
experiment, not a repeated-seed study. Existing running jobs are never stopped
by this runner.

## Random players and matches

`go random` turns the engine into a random player speaking the USI idiom:
it receives the start position plus all moves so far and answers `bestmove`.
`tools/random_match.py` is the in-between layer: it spawns two random players
(separate processes, independently seeded), forwards each move to the other
player inside the growing `position sfen ... moves ...` input, validates every
move with a third engine instance, detects the game end (royals lost, bare
king, stalemate, insufficient material, fourfold repetition as a draw, and
the 1000-ply cap as a draw), and writes one game record per game in the same
SFEN/USI style as the downloaded records:

    make random-games GAMES=10 SEED=100   # -> random_games/games/rrNNNNNN/
                                          #    {moves.usi, positions.sfen,
                                          #     positions.tsv, record.json}

Same seed reproduces the same games. The records are fully compatible with
the test suites (e.g. `python3 tests/replay_test.py random_games/games`).

## Tsume (checkmate problem) records

`tools/extract_tsume.py` converts a problem page from chushogi-renmei.com,
asahi-net.or.jp, or toybox.a.la9.jp into a JSON record holding the position
in SFEN, the published solution and variations in USI (variations carry a
`full_line` resolved through the pages' `bunki` branch tables), the authors'
move comments, and — with `--binary` — an engine validation block (SFEN
round-trip, ply-by-ply legality of every published line, and the BNS
solver's verdict/PV). Supported page families: dhtmlcb2.js (`te[]` hex
moves, KC_IPPOME/KC_NIHOME lion steps, bunki branch tables) and chush2.js
(`sk[]` string moves, 'i'/'n' lion step pairs, `skSentaku` multi-lines,
external `kifu/*.txt` data files, `--answers` for separate answer files):

    python3 tools/extract_tsume.py <problem-page-url> --binary ./chushogi-gen \
        --outdir tests/tsume
    python3 tools/mate_batch.py tests/tsume --binary ./chushogi-gen  # solve all

Records live in `tests/tsume/<id>.json` and are exercised by
`tests/mate_test.py` (every published line must be legal; problems the
solver proved must stay provable). The collection: 231 problems from the
中象戯作物図五十, 中将棋絹篩, 中将棊作物収録作品, and asahi-net
詰め中将棋 series — 187 proved mate by BNS, 22 disproved (mostly problems
whose own pages note the original text is broken or reconstructed, plus a
few genuinely cooked ones confirmed by independent brute force), 21 unknown
(very deep lines beyond the node cap; two have multi-royal defenders,
outside the solver model by design), one retro-analysis problem (tumechu78:
gote to move). Notable: tukurimono20 has no published solution line (the
page author could not transcribe the original's numbers) — the solver found
the mate on its own; the 18 position-only tsukuri problems were likewise
solved by the solver where possible.

`tools/brute_mate.cpp` (not built by the Makefile) is an independent
exhaustive tsume prover used to cross-check the BNS solver's "no mate"
verdicts; its `--refute <sfen> <moves...>` mode lists per-move verdicts for
tracing refutations (e.g. tsukuri08's escape 2d1d!).

## Rules implemented (lishogi chu shogi)## Rules implemented (lishogi chu shogi)

- 21 piece types, 18 promoted forms; no drops (captured pieces are gone).
- Promotion zone = farthest 4 ranks; promotion allowed when entering the zone
  from outside, or when capturing while starting or landing inside the zone,
  or when a pawn/lance reaches the last rank (always optional, even there).
  Note: lishogi grants this last-rank second chance to BOTH pawns and lances
  (scalashogi `Chushogi.canPromote`: `List(Pawn, Lance)`); the JCSA / Middle
  Shogi Manual rules Wikipedia describes grant it to pawns only, making an
  unpromoted last-rank lance a dead piece. Go-betweens get no exception
  anywhere, since they can move backwards.
- Lion, horned falcon and soaring eagle two-step moves, including jumps over
  adjacent pieces, igui (capture and return) and jitto (pass).
- Lion-trading restrictions (checked against scalashogi's implementation and
  its own test suite):
  - a lion may always capture an adjacent lion;
  - capturing a non-adjacent lion is forbidden if the capturing lion could be
    recaptured immediately (hidden/X-ray protectors count), unless a piece
    more valuable than a pawn/go-between is captured in the same move
    (kuisoe/tsukegui);
  - a pawn or go-between defender still counts even if captured on the way;
  - after a non-lion piece captures a lion, a non-lion piece may not capture a
    lion on a different square on the next move (the square is stored in the
    third SFEN field, e.g. `... w 5c 42`). Lishogi implements the strict
    Edo-era rule: the Okazaki amendment (allowing the counter-strike against
    an unprotected lion) is NOT implemented — scalashogi's own tests forbid
    capturing an unprotected lion on another square during the ban turn.
    One lishogi quirk to be aware of: the ban filter only examines the
    destination square of each move, so a horned falcon / soaring eagle
    capturing a lion at the MID step of a two-step move (igui hit-and-run) is
    still allowed during the ban turn, even though Wikipedia's "hit-and-run"
    wording suggests it should be banned. We mirror lishogi either way.
- No check-evasion restriction: chu shogi is played until a royal (king or
  prince) is actually captured.

## JCSA rules (`rules jcsa`)

The rule set is a runtime option of the position (`Position::rules`, default
`RuleSet::Lishogi`; not part of the SFEN), followed by `generateLegal` and
`evaluateStatus`. `RuleSet::JCSA` is the Japan Chu Shogi Association's
over-the-board rules as specified in `docs/jcsa/`, whose clause IDs the code
cites. Only these rules differ from lishogi's:

- counter-strike ban (R-L4, H-3⑥): a non-lion may not take a lion only if
  that lion has a foot, i.e. one of its own pieces protects it (judged with
  the capturing piece lifted, so X-rays count; each lion separately);
- the ban also covers a kirin that took the lion and promoted on the capture
  square (lishogi exempts that square);
- the ban also covers the mid square of a horned falcon's or soaring eagle's
  double move (lishogi tests only the destination);
- lance (R-P3, R-P4): no last-rank promotion; it promotes only by entering
  the zone or capturing in it, and one that reaches the last rank unpromoted
  is a dead piece;
- bare king (R-E4, R-E6a): only a royal plus one counting piece against a
  lone royal wins; against more pieces play continues.

Stalemate (a loss), the game ending by royal capture, and the two-royals draw
are the same under both (`docs/jcsa/interpretation.md`).

## Game-end outcomes (reference; not part of move generation)

These are lishogi server/rules-engine behaviors beyond the move generator's
scope, verified against the lishogi/scalashogi sources and the downloaded
records. The bare-king / insufficient-material / royals-lost / stalemate
evaluation is also ported into this engine (`src/status.cpp`, same logic as
scalashogi's `Chushogi.status` minus the history-dependent repetition
outcomes) and queryable through the `status` command; the 1000-ply cap is a
server rule with no position-level logic, so it is covered by a record-level
test only (see `tests/status_test.py`):

- Move cap: `Game.maxPlies(chushogi) = 1000` plies (500 moves per side; 700
  for other variants). Once the cap is exceeded the server rejects further
  moves (`TooManyPlies`) and force-draws the game with no winner
  (`drawer.force -> Status.Draw`). Record `rX1lazKt` in the downloaded data
  ended at exactly 1000 plies with status "draw"; no game is longer.
- Win conditions: capture all enemy royals (`RoyalsLost`), the bare-king rule
  (`BareKing`), or stalemate. Repetition: fourfold is a draw; perpetual
  checking/attacking loses.
- Bare king (scalashogi `Chushogi.bareKing`): a side reduced to a single
  non-dead piece that is royal loses immediately when the opponent has more
  than a bare royal, has a royal not in check, and the lone royal is not
  adjacent to any opponent piece (so it cannot capture/bare right back —
  that case continues, and baring back on the next move is a draw). So
  K+gold vs K is an automatic win as long as the lone king is not adjacent
  to the gold (or the enemy king) and the winner's king is not in check.
  Pawns and go-betweens do NOT count for the winning side (K+pawn vs K is
  not a win; the pawn/GB must first promote safely — tokin/drunk elephant
  count), and dead pieces (pawn/lance stuck on the last rank) count for
  nobody. Two games in the records ended `bareKing` (rw6DEBev, zZoeJUR8);
  scalashogi's own tests include exactly the K+G vs K case
  (`Status.BareKing`, winner = the gold side).
- Insufficient material: only two non-dead pieces left, both royal, neither
  in check -> draw (e.g. bare king vs bare king, or K+dead-pawn vs K).

## Test-case file

`tests/chushogi_cases.tsv` holds every (input, output) test case extractable
from the downloaded game records: **8688 cases from 49 games** (every ply of
every game). Regenerate with `make cases`; `make test` never rewrites it.

Lishogi publishes the ingredients (a USI move list and one SFEN per ply per
game) but no ready-made input/output test-case format, so we adopt the
standard shogi **USI protocol** idiom, which covers exactly this shape: a GUI
sends `position sfen <sfen> moves ...` and the engine answers `bestmove ...`.
Each line of the file is one tab-separated pair of literal USI lines:

    position sfen <board> <turn> <lioncap|-> <num> [moves m1 ... mk]<TAB>bestmove <mk+1>

The chu shogi extensions already present in the records are kept: files 1-12
and ranks a-l, three-square tokens for lion/falcon/eagle two-step moves
(`7e7f7g`, igui `7e7f7e`), and the third SFEN field carrying the
last-lion-capture square instead of hand pieces (chu shogi has no drops).

## Tests

    make test

- `tests/replay_test.py` — replays all 49 downloaded lishogi games
  (8688 positions). Every position of every game is a test case: the recorded
  move must be in the generated legal move list, and the resulting SFEN must
  match the recorded SFEN exactly (board, turn, lion-capture field, move
  number).
- `tests/rule_test.py` — 44 crafted positions probing the rules that replays
  cannot prove (over-generation): lion immunity-from-capture cases with kings
  (R1: adjacent/protected/hidden-protector/tsukegui/pawn-and-go-between
  defenders/multi-lion exposure; R2: the recapture ban, its same-square
  exception, the absent Okazaki amendment, hit-and-run coverage incl. the
  mid-step quirk), lion/falcon/eagle two-step geometry, jitto/igui, promotion
  conditions incl. pawn/lance last-rank deferral, initial position contents.
- `tests/perft_test.py` + `tests/perft_cases.tsv` — exact legal-move counts
  for 287 positions taken from scalashogi's own test fixtures (the lishogi
  reference values), plus initial-position perfts at depth 1 (36) and
  depth 2 (1296).
- `tests/casefile_test.py` — validates `tests/chushogi_cases.tsv` end-to-end:
  every case's output move must be legal in its input position, and
  consecutive cases must chain by the played move.
- `tests/status_test.py` — game-end outcomes: crafted bare-king /
  insufficient-material / royals-lost positions (several from scalashogi's
  own tests), the final positions of all downloaded games that ended by
  bareKing or royalsLost (evaluated and compared with the recorded winner),
  and the 1000-ply draw cap (record-level check on the one capped game).
- `tests/mate_test.py` — the BNS/df-pn checkmate solver: crafted mate-in-1
  and no-mate positions, a mate-in-3 from a real game, mate-in-1 from the
  penultimate position of every royalsLost game, multi-royal-defender
  handling (unknown), BNS vs df-pn verdict agreement, and PV legality
  validation. Most of its 482 cases (456) come from the tsume problems in
  `tests/tsume/` (all but the retro-analysis one): the 187 the solver
  proved must stay provable, and every published solution line and
  variation must replay legally.
- `tests/jcsa_test.py` — the 31 spec cases of `docs/jcsa/cases.tsv` (read
  directly) plus 23 cases for readings they do not exercise, each checked
  under both rule sets against its lishogi and JCSA verdict.

`tools/perft.cpp` (not built by the Makefile) counts perft nodes and speed;
it uses only the v0.1.0 API, so the same file builds against older versions
for regression comparisons.

All suites pass: 49/49 games replayed, 44/44 rule cases, 108/108 JCSA
cases, 289/289 perft counts, 38/38 status cases, 482/482 mate cases,
8688/8688 test cases.

### Continue the selected 1024-simulation agent

`./resume_1024_training.sh` creates a new, isolated continuation from
`simulation_budget_experiment/sims_1024/ckpt/iter_0172.pt` (step 34,600),
including its optimizer state and replay through iteration 172. It runs 20
iterations (88 games and 200 updates each), ending at step 38,600. Settings:
1024 simulations, root policy temperature 1.25, full-game sampling, 44 workers,
four inference servers, batching limit 128, and the existing mate searches and
learning-rate schedule. The source run is preserved.

Outputs go to `az/data/training_1024_34600_to38600/main/`; do not launch twice.
After training exits, a watcher automatically runs a 50-pair checkpoint
tournament at 1024 simulations, with mate search enabled, comparing steps
34,600, 36,600 and 38,600. Tournament outputs are in the parent `final_strength/`
directory; `pilot_status.json` records the watcher status. The training estimate
is about 6.5–7 hours, plus additional time for the tournament.

### Compare game-outcome, soft-Z and A0GB value targets

`./compare_value_targets.sh` launches a sequential continuation pilot from
`training_1024_34600_to38600/main/ckpt/iter_0182.pt` (step 36,600). All three
branches retain the source network and AdamW state. The experiment first
collects 88 common self-play games without training, then gives each branch
identical copies of that fresh replay. Historical replay is excluded because
older A0GB extraction stopped before entering the final searched leaf; the
experiment binary fixes this and tests one/two-ply and terminal perspectives.
Original training data and checkpoints are preserved.

Each branch runs 20 iterations, 88 games and 200 updates per iteration, ending
at step 40,600. Only `--value-target z|softz|a0gb` differs: 1024 simulations,
44 workers, four servers, root policy temperature 1.25, full-game sampling,
mate searches and the original learning-rate schedule are shared. Each branch
uses the full machine sequentially in a fixed shuffled order. Python code and
the engine are frozen with hashes before launch. Allow roughly 21–24 hours for
training and common replay collection, plus the final tournament.

Here `z` uses final-game WDL; `softz` uses root mean search score Q;
`a0gb` follows maximum-visit edges to the searched endpoint and uses its neural
or terminal score, converted to root perspective. The two scalar targets are
encoded as `[v, 0, 1-v]`; they do not retain explicit draw probability. This
compares the repository's target implementations, from a network previously
trained on z, using one training seed. The shared fresh replay start differs
from ordinary continuation using the full historical buffer. Raw loss values
across target types are not directly comparable measures of playing strength.

Monitor `az/data/value_target_experiment/status.json` and `supervisor.log`.
Branch logs and checkpoints appear under `branches/{z,softz,a0gb}/`. Each branch
keeps its midpoint and final checkpoints. On successful completion, the
supervisor runs 50 color-paired openings per matchup between the anchor and
three final checkpoints (600 games total), with fresh seed 20261105, 1024
simulations and mate search enabled. Results go to `branches/final_strength/`.
Existing output directories are rejected to prevent duplicate launches.

### Compare AdamW with hybrid Muon/AdamW

`./compare_optimizers.sh` launches a frozen, sequential continuation experiment
from the A0GB step-40,600 checkpoint (`value_target_experiment/branches/a0gb`,
iteration 202). A short pilot trains AdamW and hybrid candidates for 100 updates
each on a fixed replay subset. Muon LR multipliers are 0.3, 1 and 3; the pilot
uses the same minibatch sequence, game-disjoint train/validation splits, native
`torch.optim.Muon(adjust_lr_fn="match_rms_adamw")`, weight decay 0.01, and a
50-update Muon-only warmup. These validation games are existing replay data,
not independent unseen games. Choose the lowest finite validation loss, prefer
scale 1 within 1% of the best, and abort if all stable Muon candidates degrade
more than 20% relative to the baseline. Pilot checkpoints are never used as
starting weights for the full experiment.

After the pilot, both branches start from byte-identical original checkpoints
and replay, retain A0GB targets and the existing learning-rate schedule, and run
20 iterations each (40,600 to 44,600 updates) with 1024 simulations, 44 workers,
four eval servers, 88 games and 200 optimizer steps per iteration. AdamW retains
all moments; the hybrid retains AdamW moments for auxiliary parameters and
starts fresh Muon momentum for the transformer matrices. This tests switching
an AdamW-trained network to Muon, not training with Muon from scratch.

The hybrid assigns only 2D parameters inside `trunk` to Muon; all biases,
normalization, positional parameters and input/policy/value heads stay on
AdamW. Both optimizers step after one backward pass and a shared gradient clip.
Hybrid checkpoints save both states, parameter names, update counts and the
Muon LR/warmup settings; resume validates these. Legacy AdamW checkpoints
remain compatible. Nonfinite losses or gradient norms abort training.

Monitor `az/data/optimizer_experiment/status.json`, `pilot/results.json`, and
`branches/{adamw,muon_adamw}/console.log`. On successful completion, a fresh-seed
300-game tournament compares the anchor and both final checkpoints, with mate
search enabled at 1024 simulations. Outputs are under `branches/final_strength/`.
The source A0GB run is preserved. Frozen Python and engine hashes, all commands,
pilot selection and branch order are recorded. Allow roughly 18–22 hours for
training plus the pilot and final tournament; the selected optimizer may
change subsequent game lengths. Equal update/game counts do not imply equal
wall-clock budgets.

The completed optimizer pilot selected Muon LR scale 1.0. At step 44,600, the
hybrid scored 60% against AdamW over 100 games. A separate confirmation with
200 fresh color-paired openings (400 games) scored 58.25% (222 wins, 22 draws,
156 losses); its paired-bootstrap 95% interval was 53.75–62.75%. The hybrid is
the leading evaluated checkpoint for this continuation; this does not establish
an optimizer advantage across training seeds. Both checkpoints use the same
`ChuNet` architecture in `az/model.py`. The leading checkpoint is
`az/data/optimizer_experiment/branches/muon_adamw/ckpt/iter_0222.pt`; generated
checkpoints and game records are excluded from git.

Use `./confirm_optimizers.sh` for the 400-game confirmation, or `--check` to
inspect the command without launching it. The script preserves the original
experiment's frozen engine and tournament code, uses opening seed 20261107,
rejects existing output directories and competing workloads, saves each game,
and writes `paired_analysis.json` with an opening-pair bootstrap interval.
`./confirm_checkpoints.sh` similarly runs the earlier five-checkpoint
confirmation (2,000 games). Both scripts run in the foreground and save console
logs beside their output directories.
