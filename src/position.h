// Internal position representation for Chu Shogi, and move application.

#pragma once

#include <array>
#include <cstdint>
#include <string>

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

  // Side-to-move canonicalization: mirrored board + swapped colors, so the
  // side to move is always viewed as sente (see flipSquare below).
  Position flipped() const;
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

// ---- Side-to-move canonicalization (AlphaZero input convention) ----
// Mirror the board across the horizontal axis and swap colors, so the side
// to move is always encoded as sente (JHBR3's board.Flipped() equivalent).
inline int flipSquare(int sq) {
  return (kRanks - 1 - sqY(sq)) * kFiles + sqX(sq);
}
inline uint8_t flipPiece(uint8_t p) { return p ? ((p - 1) ^ 1) + 1 : 0; }
inline Move flipMove(const Move& m) {
  Move r = m;
  r.from = flipSquare(m.from);
  r.to = flipSquare(m.to);
  if (m.mid != kNoSquare) r.mid = flipSquare(m.mid);
  return r;
}
// Same transform on the move as a USI string (e.g. "6c6d" -> "6j6i").
std::string flipUsiMove(const std::string& token);

}  // namespace chu
