// Command parsing shared by the engines' USI-Chu mode (USI-CHU.md §3).
// Unlike the line protocol's commands, a bad command is reported, never
// fatal.

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "movegen.h"
#include "position.h"
#include "sfen.h"
#include "usi.h"

namespace chu {

inline constexpr const char* kStartSfen =
    "lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/"
    "3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1";

// position startpos | sfen <board> <b|w> <lioncap|-> <num> [moves <m1> ...]
// Sets `pos` from `tok` (the command split on spaces) under `rules`.
// `onMove(before, move)` runs before each move is applied. False and `err`
// on a malformed command, a bad SFEN or an illegal move; `pos` is then
// unspecified.
inline bool parseUsiPosition(
    const std::vector<std::string>& tok, RuleSet rules, Position& pos,
    std::string& err,
    const std::function<void(const Position&, const Move&)>& onMove = {}) {
  size_t i = 2;
  std::string sfen;
  if (tok.size() >= 2 && tok[1] == "startpos") {
    sfen = kStartSfen;
  } else if (tok.size() >= 6 && tok[1] == "sfen") {
    sfen = tok[2] + " " + tok[3] + " " + tok[4] + " " + tok[5];
    i = 6;
  } else {
    err = "expected: position startpos|sfen <sfen> [moves ...]";
    return false;
  }
  pos.rules = rules;
  if (!parseSfen(sfen, pos)) {
    err = "bad SFEN: " + sfen;
    return false;
  }
  pos.rules = rules;
  if (i < tok.size() && tok[i] != "moves") {
    err = "expected 'moves' after the position";
    return false;
  }
  for (++i; i < tok.size(); ++i) {
    Move m;
    if (!parseUsiMove(tok[i], m) || !isLegal(pos, m)) {
      err = "illegal move " + tok[i] + " in " + toSfen(pos);
      return false;
    }
    if (onMove) onMove(pos, m);
    pos.apply(m);
  }
  return true;
}

// setoption name <name> [value <value>]; names and values may hold spaces.
inline bool parseSetOption(const std::vector<std::string>& tok,
                           std::string& name, std::string& value) {
  name.clear();
  value.clear();
  if (tok.size() < 3 || tok[1] != "name") return false;
  std::string* out = &name;
  for (size_t i = 2; i < tok.size(); ++i) {
    if (tok[i] == "value" && out == &name) {
      out = &value;
      continue;
    }
    if (!out->empty()) *out += ' ';
    *out += tok[i];
  }
  return !name.empty();
}

// The `id <n>` of a `go` command (USI-CHU.md §3), or "" without one.
inline std::string goId(const std::vector<std::string>& tok) {
  for (size_t i = 1; i + 1 < tok.size(); ++i)
    if (tok[i] == "id") return tok[i + 1];
  return "";
}

}  // namespace chu
