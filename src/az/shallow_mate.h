// Fixed-depth checking search shared by USI root and leaf protection.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <unordered_set>
#include "../mate.h"
#include "repetition_guard.h"

namespace az {

inline int oddMateDepth(int depth) {
  depth = std::clamp(depth, 0, 7);
  return depth > 0 && depth % 2 == 0 ? depth - 1 : depth;
}

struct ShallowMateLimits {
  size_t nodeLimit = 20000;
  size_t nodes = 0;
  const std::atomic<bool>* stop = nullptr;
  std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::time_point::max();
  bool cancelled = false;
  bool poll();
};

// Exact repetition adjudication from the supplied actual-game history.
chu::MateAnswer probeShallowMate(const chu::Position& root,
    const RepetitionHistory& history, int depth, ShallowMateLimits& limits,
    int maxPlyCap = 1000);

// MCTS has canonical board keys, not perpetual-attack counters. Treat every
// repeated board on a proof branch as unknown. This conservatively avoids
// proving mates through historical draws/fouls. Prefix excludes root itself.
chu::MateAnswer probeShallowMate(const chu::Position& root,
    const std::vector<uint64_t>& previousCanonicalKeys, int depth,
    ShallowMateLimits& limits, int maxPlyCap = 1000);

// Reuse MCTS's game-prefix membership table across all leaf probes. The
// search-path view excludes the current leaf, without copying the prefix.
struct ShallowMateHistory {
  const std::unordered_set<uint64_t>* game = nullptr;
  const std::vector<uint64_t>* path = nullptr;
  size_t pathLength = 0;
};
chu::MateAnswer probeShallowMate(const chu::Position& root,
    const ShallowMateHistory& previous, int depth, ShallowMateLimits& limits,
    int maxPlyCap = 1000);

}  // namespace az
