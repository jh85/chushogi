// Conversion layer: USI move text <-> internal Move.
//
// Chu shogi USI extends the standard format:
//   ordinary move:        <from><to>["+"], e.g. 8i8h, 12h12i+
//   lion-power two-step:  <from><mid><to>, e.g. 7e7f7g; igui/jitto return to
//                         the origin, e.g. 7e7f7e
// Squares are <file 1-12><rank a-l>. A trailing '=' or '?' (declined/
// uncertain promotion annotations) is accepted and treated as no promotion.

#pragma once

#include <string>

#include "position.h"

namespace chu {

std::string squareKey(int sq);
bool parseSquare(const std::string& s, int& out);

std::string moveToUsi(const Move& m);
bool parseUsiMove(const std::string& token, Move& out);

}  // namespace chu
