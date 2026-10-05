// Internal position representation for Chu Shogi, and move application.

#pragma once

#include <array>
#include <cstdint>

#include "types.h"

namespace chu {

struct Position {
  std::array<uint8_t, kSquares> board{};  // piece ids, 0 = empty
  Color sideToMove = Sente;
  // Chu shogi lion-trading state: if the immediately preceding move captured
  // a lion with a non-lion piece, this holds the square where that lion was
  // taken; otherwise kNoSquare. (Serialized in the third SFEN field.)
  int16_t lastLionCapture = kNoSquare;
  int moveNumber = 1;

  uint8_t at(int sq) const { return board[sq]; }
  bool empty(int sq) const { return board[sq] == 0; }

  // Applies a move (assumed legal, e.g. one of generateLegal()'s results).
  // Mirrors scalashogi Situation.move + finalizeAfterUsi.
  void apply(const Move& m);
};

// Promotion-zone helpers (Chushogi.promotionRanks / backrank).
inline bool inPromotionZone(int sq, Color c) {
  int y = sqY(sq);
  return c == Sente ? y <= 3 : y >= 8;
}
inline int backrankY(Color c) { return c == Sente ? 0 : kRanks - 1; }

// Chushogi.canPromote: promotion is possible when entering the zone from
// outside, or when making a capture while starting or landing in the zone,
// or when a pawn/lance reaches the last rank (even inside the zone).
inline bool canPromote(Role role, Color c, int from, int to, bool capture) {
  if (promoteRole(role) < 0) return false;
  bool destIn = inPromotionZone(to, c);
  bool origIn = inPromotionZone(from, c);
  if (destIn && !origIn) return true;
  if (capture && (destIn || origIn)) return true;
  if ((role == Pawn || role == Lance) && sqY(to) == backrankY(c)) return true;
  return false;
}

}  // namespace chu
