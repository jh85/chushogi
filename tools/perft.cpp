// Perft: counts the leaves of the legal-move tree to a fixed depth, for
// move-generator regression checks (node counts) and speed measurements.
//
// Not built by the Makefile. It uses only the v0.1.0 API, so the same file
// builds against any tagged version for before/after comparisons:
//   g++ -O2 -std=c++17 -Isrc src/{types,position,sfen,usi,movegen}.cpp tools/perft.cpp -o perft
//
// Usage: perft <depth> < positions
//   One SFEN per line; anything after a tab is ignored, so
//   tests/perft_cases.tsv can be fed as is. Prints "<nodes>\t<sfen>" per
//   position on stdout, and the total and the time spent on stderr. The
//   positions use the default rule set (lishogi).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "movegen.h"
#include "sfen.h"

namespace {

uint64_t perft(const chu::Position& pos, int depth) {
  const std::vector<chu::Move> moves = chu::generateLegal(pos);
  if (depth == 1) return moves.size();
  uint64_t nodes = 0;
  for (const chu::Move& m : moves) {
    chu::Position child = pos;
    child.apply(m);
    nodes += perft(child, depth - 1);
  }
  return nodes;
}

}  // namespace

int main(int argc, char** argv) {
  const int depth = argc == 2 ? std::atoi(argv[1]) : 0;
  if (depth < 1) {
    std::cerr << "usage: perft <depth> < positions\n";
    return 2;
  }
  std::vector<std::string> sfens;
  for (std::string line; std::getline(std::cin, line);) {
    line = line.substr(0, line.find('\t'));
    if (!line.empty()) sfens.push_back(line);
  }

  uint64_t total = 0;
  const auto start = std::chrono::steady_clock::now();
  for (const std::string& sfen : sfens) {
    chu::Position pos;
    if (!chu::parseSfen(sfen, pos)) {
      std::cerr << "bad SFEN: " << sfen << "\n";
      return 1;
    }
    const uint64_t nodes = perft(pos, depth);
    total += nodes;
    std::cout << nodes << '\t' << sfen << '\n';
  }
  const double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - start)
                          .count();
  std::fprintf(stderr, "%zu positions, depth %d: %llu nodes in %.3f s "
               "(%.2f Mnodes/s)\n", sfens.size(), depth,
               static_cast<unsigned long long>(total), secs,
               total / secs / 1e6);
  return 0;
}
