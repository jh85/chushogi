// Shared attack detection (scalashogi posThreatened equivalent), used by the
// move generator and by game-end status evaluation.

#pragma once

#include "position.h"

namespace chu {

// Squares rem1/rem2 (kNoSquare allowed) are treated as empty, mirroring
// scalashogi's Board.forceTake in the lion-capture threat tests.
inline bool squareEmpty(const Position& pos, int sq, int rem1, int rem2) {
  return sq == rem1 || sq == rem2 || pos.board[sq] == 0;
}

inline Dir physical(Dir d, Color c) {
  return {d.dx, static_cast<int8_t>(c == Sente ? d.dy : -d.dy)};
}

// Can the piece of color `by` on `from` capture on `to`, ignoring the mover's
// own safety? Jumps and steps ignore intervening pieces; sliders need a clear
// path. Equivalent to the per-piece test in scalashogi's posThreatened.
inline bool pieceAttacks(const Position& pos, int from, int to, Color by,
                         int rem1, int rem2) {
  const int fx = sqX(from), fy = sqY(from);
  const int tx = sqX(to), ty = sqY(to);
  const int dx = tx - fx, dy = ty - fy;
  const RoleMoves& rm = roleMoves()[roleOf(pos.board[from])];

  for (Dir d : rm.direct) {
    Dir p = physical(d, by);
    if (dx == p.dx && dy == p.dy) return true;
  }
  for (Dir d : rm.proj) {
    Dir u = physical(d, by);
    if (dx * u.dy != dy * u.dx) continue;             // not colinear
    if (dx * u.dx + dy * u.dy <= 0) continue;         // wrong way / same square
    int cx = fx + u.dx, cy = fy + u.dy;
    bool blocked = false;
    while (cx != tx || cy != ty) {
      if (!squareEmpty(pos, makeSq(cx, cy), rem1, rem2)) {
        blocked = true;
        break;
      }
      cx += u.dx;
      cy += u.dy;
    }
    if (!blocked) return true;
  }
  return false;
}

// posThreatened: is `target` attacked by any piece of color `by`?
// pawnsOnly restricts attackers to pawns and go-betweens (the lion-trading
// rule for "insignificant" defenders).
inline bool threatened(const Position& pos, Color by, int target,
                       bool pawnsOnly, int rem1, int rem2) {
  for (int f = 0; f < kSquares; ++f) {
    if (f == rem1 || f == rem2) continue;
    uint8_t p = pos.board[f];
    if (!p || colorOf(p) != by) continue;
    Role r = roleOf(p);
    if (pawnsOnly && r != Pawn && r != GoBetween) continue;
    if (pieceAttacks(pos, f, target, by, rem1, rem2)) return true;
  }
  return false;
}

// Variant.check: is any royal (king or prince) of `color` under attack?
inline bool inCheck(const Position& pos, Color color) {
  for (int s = 0; s < kSquares; ++s) {
    uint8_t p = pos.board[s];
    if (p && colorOf(p) == color && isRoyal(roleOf(p)) &&
        threatened(pos, !color, s, /*pawnsOnly=*/false, kNoSquare, kNoSquare))
      return true;
  }
  return false;
}

}  // namespace chu
