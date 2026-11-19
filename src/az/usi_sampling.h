// Visit-based opening sampling in the root's canonical board frame.
#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "mcts.h"

namespace az {

// unsafeRootMoves must be populated by MctsConfig::safeSampling. If no safe
// move has visits, keep the proof/royal-aware bestMove rather than sample an
// unvisited move. Forced mates and the all-lost slowest-loss pick take priority.
inline chu::Move selectUsiMove(const SearchResult& search, int samplingPlies,
                               int ply, double temperature,
                               std::mt19937_64& rng) {
  if (ply >= samplingPlies || samplingPlies <= 0 ||
      !std::isfinite(temperature) || temperature <= 0 ||
      search.provenMate || search.allRootMovesLost)
    return search.bestMove;

  std::vector<int> indices;
  std::vector<double> weights;
  double maxVisit = 0;
  for (const auto& [index, visits] : search.policy) {
    if (!std::isfinite(visits) || visits <= 0 ||
        std::find(search.unsafeRootMoves.begin(), search.unsafeRootMoves.end(),
                  index) != search.unsafeRootMoves.end())
      continue;
    indices.push_back(index);
    weights.push_back(visits);
    maxVisit = std::max(maxVisit, static_cast<double>(visits));
  }
  if (indices.empty()) return search.bestMove;
  if (indices.size() == 1) return indexToMove(indices.front());

  // N^(1/T), scaled in log space: stable even for very small temperatures.
  const double logMax = std::log(maxVisit);
  for (double& weight : weights)
    weight = std::exp((std::log(weight) - logMax) / temperature);
  std::discrete_distribution<size_t> pick(weights.begin(), weights.end());
  return indexToMove(indices[pick(rng)]);
}

}  // namespace az
