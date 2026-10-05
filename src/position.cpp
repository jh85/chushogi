#include "position.h"

namespace chu {

void Position::apply(const Move& m) {
  const uint8_t mover = board[m.from];
  const Role moverRole = roleOf(mover);
  // Captured pieces as they were before the move (needed for the lion-capture
  // history update below).
  const uint8_t oldAtTo = board[m.to];
  const uint8_t oldAtMid = m.mid != kNoSquare ? board[m.mid] : 0;

  uint8_t placed = mover;
  if (m.promote) placed = makePiece(static_cast<Role>(promoteRole(moverRole)),
                                    colorOf(mover));

  board[m.from] = 0;
  board[m.to] = placed;  // from == to for igui/jitto: no-op, as intended
  if (m.mid != kNoSquare) board[m.mid] = 0;

  // lastLionCapture: set iff a non-lion just captured an enemy lion; the
  // square is searched starting from the final destination, then the mid step
  // (scalashogi checks usi.positions.drop(1).reverse).
  lastLionCapture = kNoSquare;
  if (!isLion(moverRole)) {
    const Color enemy = !colorOf(mover);
    if (oldAtTo && colorOf(oldAtTo) == enemy && isLion(roleOf(oldAtTo)))
      lastLionCapture = m.to;
    else if (m.mid != kNoSquare && oldAtMid && colorOf(oldAtMid) == enemy &&
             isLion(roleOf(oldAtMid)))
      lastLionCapture = m.mid;
  }

  sideToMove = !sideToMove;
  ++moveNumber;
}

}  // namespace chu
