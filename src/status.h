// Game-end status evaluation, ported from scalashogi's Chushogi.status:
//   RoyalsLost  - side to move has no royal piece left
//   BareKing    - one side was reduced to a lone royal under the baring rule
//   Stalemate   - side to move has no legal moves (loses)
//   Draw        - insufficient material (only two royals, no checks)
// Repetition/perpetual-check outcomes are history-dependent and therefore
// out of scope (they cannot be computed from a single position).

#pragma once

#include "position.h"

namespace chu {

enum class GameEnd { Playing, RoyalsLost, BareKing, Stalemate, Draw };

struct StatusResult {
  GameEnd end = GameEnd::Playing;
  int winner = -1;  // Sente(0) / Gote(1), or -1 for draws and ongoing games
};

StatusResult evaluateStatus(const Position& pos);

const char* gameEndName(GameEnd end);
const char* colorName(int color);

}  // namespace chu
