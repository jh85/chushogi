// Brute-force tsume checker: independent cross-check of the BNS mate
// solver. Exhaustive minimax over the restricted tree (attacker plays only
// checking moves or royal captures, defender plays only evasions), with a
// transposition table and route-cycle cutoff — the same model as
// mate.h, minus the BNS proof machinery. Prints the verdict and, on
// escape, a refutation line.
//
// Usage: brute_mate "<sfen>" [maxDepth]
//
// Not built by the Makefile (debug tool):
//   g++ -O2 -std=c++17 -Isrc src/types.cpp src/position.cpp src/sfen.cpp \
//       src/usi.cpp src/movegen.cpp src/status.cpp src/mate.cpp \
//       tools/brute_mate.cpp -o brute_mate

#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "attacks.h"
#include "movegen.h"
#include "position.h"
#include "sfen.h"
#include "usi.h"

using namespace chu;

namespace {

int royalCount(const Position& pos, Color c) {
  int n = 0;
  for (int s = 0; s < kSquares; ++s) {
    uint8_t p = pos.board[s];
    if (p && colorOf(p) == c && isRoyal(roleOf(p))) ++n;
  }
  return n;
}

bool royalAttacked(const Position& pos, Color c) {
  for (int s = 0; s < kSquares; ++s) {
    uint8_t p = pos.board[s];
    if (p && colorOf(p) == c && isRoyal(roleOf(p)) &&
        threatened(pos, !c, s, false, kNoSquare, kNoSquare))
      return true;
  }
  return false;
}

using Key = std::string;  // exact board state: squares + side + lion history

Key keyOf(const Position& pos) {
  Key k;
  k.reserve(kSquares + 4);
  for (int s = 0; s < kSquares; ++s) k.push_back(static_cast<char>(pos.board[s]));
  k.push_back(static_cast<char>(pos.sideToMove));
  k.push_back(static_cast<char>(pos.lastLionCapture & 0xff));
  k.push_back(static_cast<char>(pos.lastLionCapture >> 8));
  return k;
}

struct Entry {
  int depth = -1;
  bool win = false;
};

std::unordered_map<Key, Entry> tt;
std::vector<Key> path;
std::vector<Move> refutation;

// attacker-to-move = true node. Returns true if the attacker forces the
// defender's last royal within `depth` plies.
bool solve(Position& pos, bool attackerTurn, int depth) {
  const Color us = pos.sideToMove;
  const Color them = !us;
  const Key h = keyOf(pos);
  auto it = tt.find(h);
  if (it != tt.end() && it->second.depth >= depth) return it->second.win;

  for (const Key& pk : path)
    if (pk == h) return false;  // route cycle: no forced mate this way

  const std::vector<Move> legal = generateLegal(pos);
  bool win = false;
  path.push_back(h);
  struct Pop {
    ~Pop() { path.pop_back(); }
  } popper;

  if (attackerTurn) {
    for (const Move& m : legal) {
      Position c = pos;
      c.apply(m);
      if (royalCount(c, them) == 0) {  // royal captured: immediate win
        win = true;
        break;
      }
      if (!royalAttacked(c, them)) continue;  // must check
      if (depth > 0) {
        bool sub = solve(c, false, depth - 1);
        if (sub) {
          win = true;
          break;
        }
      }
    }
  } else {
    bool hasEvasion = false;
    bool allLose = true;
    std::vector<Move> refutMove;
    for (const Move& m : legal) {
      Position c = pos;
      c.apply(m);
      if (royalCount(pos, them) > 0 && royalCount(c, them) == 0)
        continue;  // attacker captured out: escape (kingless attacker: never)
      if (royalAttacked(c, us)) continue;  // still in check: not an evasion
      hasEvasion = true;
      if (depth <= 0) {
        allLose = false;
        refutMove = {m};
        break;
      }
      bool sub = solve(c, true, depth - 1);
      if (!sub) {
        allLose = false;
        refutMove = {m};
        break;
      }
    }
    if (!hasEvasion) {
      win = true;  // no way to save the royal: mate
    } else {
      win = allLose;
      if (!win && refutation.empty()) refutation = refutMove;
    }
  }

  tt[h] = {depth, win};
  return win;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: brute_mate <sfen> [maxDepth] | brute_mate --refute "
                 "<sfen> <move1> [move2 ...]   (list defender verdicts)\n";
    return 1;
  }
  bool refute = std::string(argv[1]) == "--refute";
  const char* sfenArg = argv[refute ? 2 : 1];
  int firstMove = refute ? 3 : 2;
  Position pos;
  if (!parseSfen(sfenArg, pos)) {
    std::cerr << "bad SFEN\n";
    return 1;
  }
  auto applyToken = [&](const char* tok) {
    Move m;
    if (!parseUsiMove(tok, m) || !isLegal(pos, m)) return false;
    pos.apply(m);
    return true;
  };
  if (refute) {
    // Apply the given line (from the sente-to-move problem root), then list
    // every legal move of the side to move with its tsume verdict.
    int plies = 0;
    for (int i = firstMove; i < argc; ++i, ++plies) {
      if (!applyToken(argv[i])) {
        std::cerr << "illegal: " << argv[i] << "\n";
        return 1;
      }
    }
    const bool listingAttacker = (plies % 2 == 0);  // attacker = sente
    const Color us = pos.sideToMove;
    const Color them = !us;
    for (const Move& m : generateLegal(pos)) {
      Position c = pos;
      c.apply(m);
      std::string tag;
      if (royalCount(pos, them) > 0 && royalCount(c, them) == 0)
        tag = "captures-last-royal";
      else if (listingAttacker && !royalAttacked(c, them))
        tag = "no-check";
      else if (!listingAttacker && royalAttacked(c, us))
        tag = "still-in-check";
      else {
        tt.clear();
        bool sub = solve(c, !listingAttacker, 15);
        if (listingAttacker)  // defender node must lose for the move to work
          tag = sub ? "attacker-wins" : "DEFENDER-ESCAPES";
        else  // we listed an evasion; does the attacker still force mate?
          tag = sub ? "mated-anyway" : "ESCAPE-REFUTATION";
      }
      std::cout << moveToUsi(m) << "  " << tag << "\n";
    }
    return 0;
  }
  int maxDepth = argc > 2 ? std::atoi(argv[2]) : 13;
  for (int d = 1; d <= maxDepth; d += 2) {
    tt.clear();
    path.clear();
    refutation.clear();
    bool win = solve(pos, true, d);
    std::cout << "depth " << d << ": " << (win ? "mate" : "no mate")
              << " (tt=" << tt.size() << ")\n";
    if (win) break;
  }
  return 0;
}
