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

Position Position::flipped() const {
  Position out;
  for (int s = 0; s < kSquares; ++s) out.board[flipSquare(s)] = flipPiece(board[s]);
  out.sideToMove = !sideToMove;
  out.lastLionCapture =
      lastLionCapture == kNoSquare ? kNoSquare : flipSquare(lastLionCapture);
  out.moveNumber = moveNumber;
  out.rules = rules;
  return out;
}

std::string flipUsiMove(const std::string& token) {
  std::string out;
  size_t i = 0;
  while (i < token.size()) {
    if (token[i] == '+') {
      out += '+';
      ++i;
      continue;
    }
    // one square = file digits + rank letter
    size_t j = i;
    while (j < token.size() && token[j] >= '0' && token[j] <= '9') ++j;
    if (j == i || j >= token.size()) break;
    int file = std::stoi(token.substr(i, j - i));
    char rank = token[j];
    out += std::to_string(file);
    out += static_cast<char>('a' + ('l' - rank));
    i = j + 1;
  }
  return out;
}

}  // namespace chu
