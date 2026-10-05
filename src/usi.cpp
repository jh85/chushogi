#include "usi.h"

#include <cctype>

namespace chu {

std::string squareKey(int sq) {
  std::string s = std::to_string(sqX(sq) + 1);
  s += static_cast<char>('a' + sqY(sq));
  return s;
}

bool parseSquare(const std::string& s, int& out) {
  if (s.size() < 2 || s.size() > 3) return false;
  size_t i = 0;
  int file = 0;
  while (i < s.size() && std::isdigit(s[i])) {
    file = file * 10 + (s[i] - '0');
    ++i;
  }
  if (i != s.size() - 1 || file < 1 || file > kFiles) return false;
  char r = s[i];
  if (r < 'a' || r >= 'a' + kRanks) return false;
  out = makeSq(file - 1, r - 'a');
  return true;
}

std::string moveToUsi(const Move& m) {
  std::string s = squareKey(m.from);
  if (m.mid != kNoSquare) s += squareKey(m.mid);
  s += squareKey(m.to);
  if (m.promote) s += '+';
  return s;
}

bool parseUsiMove(const std::string& token, Move& out) {
  std::string body = token;
  bool promote = false;
  if (!body.empty() && (body.back() == '+' || body.back() == '=' ||
                        body.back() == '?')) {
    promote = body.back() == '+';
    body.pop_back();
  }
  if (body.empty() || body.back() < 'a' || body.back() > 'l') return false;

  int squares[3];
  int n = 0;
  size_t i = 0;
  while (i < body.size()) {
    if (n == 3 || !std::isdigit(body[i])) return false;
    size_t start = i;
    while (i < body.size() && std::isdigit(body[i])) ++i;
    if (i >= body.size() || body[i] < 'a' || body[i] > 'l') return false;
    int file = std::stoi(body.substr(start, i - start));
    if (file < 1 || file > kFiles) return false;
    squares[n++] = makeSq(file - 1, body[i] - 'a');
    ++i;
  }
  if (n != 2 && n != 3) return false;

  out.from = static_cast<int16_t>(squares[0]);
  if (n == 3) {
    out.mid = static_cast<int16_t>(squares[1]);
    out.to = static_cast<int16_t>(squares[2]);
  } else {
    out.mid = kNoSquare;
    out.to = static_cast<int16_t>(squares[1]);
  }
  out.promote = promote;
  return true;
}

}  // namespace chu
