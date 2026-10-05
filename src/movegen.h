// Chu Shogi legal move generator.
//
// Produces every move lishogi/scalashogi would accept for the chu shogi
// variant:
//   - all pseudo-legal moves (chu shogi has no check-evasion restriction: a
//     player "need not move out of check", royals are actually captured),
//   - optional promotions (entering the zone, capturing in/out of the zone,
//     pawn/lance reaching the last rank),
//   - lion / horned falcon / soaring eagle two-step moves, including igui
//     (capture and return) and jitto (pass),
//   - lion-trading restrictions:
//       * a lion may not capture a non-adjacent enemy lion if it could be
//         recaptured right after (hidden protectors and X-rays count),
//         unless the same move also captures a piece more valuable than a
//         pawn/go-between (kuisoe/tsukegui);
//       * a pawn or go-between defending a lion still counts as defending it
//         even if the lion move captures that defender on the way;
//       * after a non-lion piece captured an enemy lion, a non-lion piece may
//         not capture a lion on a different square on the following move.

#pragma once

#include <vector>

#include "position.h"

namespace chu {

// All legal moves for the side to move.
std::vector<Move> generateLegal(const Position& pos);

// True if `m` is among the legal moves (used when replaying ordered USI move
// lists through the conversion layer).
bool isLegal(const Position& pos, const Move& m);

}  // namespace chu
