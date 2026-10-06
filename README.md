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

BNS vs df-pn on the tsume collection (tools/mate_compare.py, 207 provable
problems, identical node caps): verdicts never contradict; on the 174
positions both prove, df-pn spends 1.13x BNS's nodes; on the 21 disproofs
df-pn is faster (0.76x). But df-pn fails to prove 12 positions within 2x
the node count BNS needs (they come back "unknown") — BNS's branch-number
thresholds are markedly better at driving deep proofs to completion, which
is why it is the default.


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
az_pipe_test (untrained net end-to-end).

``` 
make chushogi-az
python3 az/run_loop.py --workers 8 --games 2 --sims 64 --iterations 300
```

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
every game). Regenerate with `make tests/chushogi_cases.tsv` (or delete and
run `make test`).

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
  validation.

All suites pass: 49/49 games replayed, 44/44 rule cases, 289/289 perft
counts, 34/34 status cases, 26/26 mate cases, 8688/8688 test cases.
