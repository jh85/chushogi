# Root royal safety and proof-aware selection

This change was developed against `origin/usi` at `40b6a99`, based on the
supplied `az-blunder-4j5i-root-royal-filter.patch`, then merged into `main`
alongside the existing hybrid Muon/AdamW optimizer integration. Running jobs
use frozen source and engine snapshots and are not modified by this merge.

## USI play

USI enables both `rootProofSelection` and `rootRoyalFilter`. Root children
proved to win for the opponent are skipped while an eligible alternative
remains. A node stores `MateAnswer::matePly`. If all candidates are lost,
selection prefers the longest returned proof line, then visits. This is a
mate-line-length estimate, **not an exact minimax distance-to-loss solver**:
the existing tsume solver need not return the longest defender line and may
end its PV at no-evasion rather than actual royal capture.

The supplied geometric `hangsLastRoyal` backstop is retained, including lion
mid-square captures and king-plus-prince handling. It also checks adjudication
so a move that already wins or draws is not rejected for a hypothetical reply.
Known terminal draws and the game ply cap are respected. Under a small budget,
unexpanded hanging moves can be excluded without a mate proof. Unknown probe
verdicts never count as proofs.

CLI reproduction uses `--proof-aware 1 --royal-filter 1`; these are opt-in for
`go az`/`analyze az` and enabled in USI only. Analysis includes `best_move`,
`proven_loss`, `loss_plies`, policy targets and selection diagnostics. Search
visits and targets are unchanged by USI selection alone. `info string` logs
report effective search settings and any safety override.

With the provided full 318-ply history, soft-Z iteration 202 and the reported
1024/16/5000/3/2000 settings, legacy selection reproduces `4j5i`; proof-aware
selection returns `4j5j` with identical move visit counts.

## Retrospective measurement before changing training

`az/royal_safety_audit.py` selects 20 games per stratum, then samples 5,000
positions uniformly within that position pool, using fixed seed 20261108.
Three self-play strata cover soft-Z/A0GB step 40,600 and hybrid step 64,600;
the hybrid stratum is restricted to iterations 303–322. The soft-Z/A0GB strata
include their common initial replay batch. These are run-level samples, not
only final-checkpoint-generated positions. The evaluation stratum covers the
step-44,600/60,600/64,600 tournament. All 320
positions in the supplied rated game are audited separately.

A qualifying alternative is legal, nonhanging, and not proved lost by a fresh
depth-3/2000-node probe. It need not be winning. Proof flags and original TT
states are absent from old records; **re-proved** losses are retrospective,
not a count of historical search verdicts. The audit uses no neural inference.

| Stratum | Positions | Played hanging, alternative exists | Played re-proved loss, alternative exists | Individual unsafe policy entry >=10%, alternative exists |
|---|---:|---:|---:|---:|
| Soft-Z self-play | 5,000 | 11 (0.22%) | 11 (0.22%) | 0 |
| A0GB self-play | 5,000 | 12 (0.24%) | 13 (0.26%) | 2 (0.04%) |
| Hybrid self-play | 5,000 | 7 (0.14%) | 9 (0.18%) | 1 (0.02%) |
| Recent evaluation, mixed checkpoints | 5,000 | 3 (0.06%) | 7 (0.14%) | unavailable |
| Supplied rated game | 320 | 1 | 1 | unavailable |

Combined self-play: 30/15,000 hanging moves (0.20%), 33/15,000 re-proved losses
(0.22%), and 3/15,000 positions with an individual >=10% unsafe policy entry
(0.02%), all conditional on an unproved nonhanging alternative existing.
The loss column includes immediate terminal adjudications and re-proved mates.
These are sampled empirical rates, not a census or independent-position
confidence calculation. Smaller policy entries are not probed unless played
or visit-argmax; these rates do not estimate all unsafe policy mass.
Evaluation and gate records do not contain visit distributions. Near the
1000-ply cap, proofs are conservatively restricted to available continuation.
The exact supplied blunder is recovered at zero-based ply 318.

Artifacts in the training checkout:
`az/data/royal_safety_audit_20261108_final/{summary.json,selection.json,positions.jsonl}`.
Neural reproduction and CPU fork smoke results are under
`az/data/royal_safety_audit_20261108_full/`.

## Experimental self-play flags (both off by default)

* Native `selfplay --safe-sampling 1`: exclude proved-lost and hanging moves
  when a nonexcluded alternative exists, for both sampling and post-temperature
  greedy choice. If eligible visits total zero, sample uniformly among eligible
  moves. It leaves stored policy targets unchanged unless separately requested.
* Native `selfplay --safe-policy 1`: zero those unsafe target entries and
  renormalize the remaining visit mass. If eligible mass is zero, use a uniform
  eligible target. Actual MCTS visits, rootQ and A0GB backup are not altered.
* If every move is lost, retain the original legal distribution; never emit an
  all-zero target. The independent USI selector can still prefer a longer loss.
* `run_loop.py --safe-sampling --safe-policy` forwards these flags only when
  enabled. Legacy binaries/default jobs receive no extra native options.
* Records declare both settings in their `selfplay` metadata. No live job is
  switched to the experimental behavior.

The already-existing A0GB endpoint fix from `main` is retained here, since the
USI branch predates it. No new value-target algorithm is introduced.

## Checks

`make az-root-safety-test` tests the real hanging position under an adversarial
prior, non-immediate mate proof, all-lost length preference, disabled-probe
backstop, terminal wins/draws, kingless tsume, prince survival, lion double-royal
capture, zero eligible visit mass, target normalization and unchanged default
visits/targets. The original `az_blunder_4j5i` USI test covers both colors.
Existing search-value, sampling, MCTS, probe, legal-policy and self-play tests
also pass.

An isolated CPU-only fork smoke uses the same step-64,600 network for baseline
and flagged self-play, then resumes each clone for one optimizer update. This
checks record/target compatibility and preservation of both hybrid optimizer
states; it is not a strength experiment. A long paired training comparison
should wait until the current live continuation completes and should keep
these changes opt-in until evaluated.

The CPU smoke used 16 simulations per ply and completed one baseline game (203 plies) and one flagged game
(793 plies), each followed by one successful hybrid update to step 64,601,
with Muon update count 24,001. One game per configuration cannot establish
strength or throughput; the length difference is a reason to measure both in
a real controlled training fork. No GPU inference or live training changes
were used for the audit or smoke.

The original USI branch predates main's Muon optimizer integration; the merge
retains both the optimizer options and the independent safety flags. The CPU
smoke intentionally used main's trainer. This change does not replace the
active production trainer or launch a long training experiment.
