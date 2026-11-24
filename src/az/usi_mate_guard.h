// Bounded, history-aware checking search for actual USI play.
#pragma once

#include "mcts.h"
#include "repetition_guard.h"
#include "shallow_mate.h"

namespace az {

using MateGuardLimits = ShallowMateLimits;

// The side to move attacks, always checking. Only kMate is a proof. Odd
// depths count through the final mating check, like JHBR3. Since chu shogi
// permits non-evasions, also verify the subsequent actual royal capture,
// including history and the game cap. Multi-royal defenders are unknown.
chu::MateAnswer probeUsiMate(const chu::Position& root,
                           const RepetitionHistory& history, int depth,
                           MateGuardLimits& limits, int maxPlyCap = 1000);

struct RootMateGuardResult {
  chu::Move move;  // canonical root frame, like SearchResult
  std::vector<chu::Move> rejected;
  bool allLost = false;
  int lossPlies = -1;  // proof-line proxy, not exact minimax distance
};

// Check the actual sampled/greedy choice first; after a proof, try the next
// eligible move by visits. Unknown/cancelled probes never reject a move.
RootMateGuardResult guardUsiMove(const chu::Position& root,
    const RepetitionHistory& history, const SearchResult& search,
    chu::Move selected, int depth, MateGuardLimits& limits);

}  // namespace az
