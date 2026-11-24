#include "usi_mate_guard.h"

namespace az {

chu::MateAnswer probeUsiMate(const chu::Position& root,
    const RepetitionHistory& history, int depth, MateGuardLimits& limits, int cap) {
  return probeShallowMate(root, history, depth, limits, cap);
}

RootMateGuardResult guardUsiMove(const chu::Position& root,
    const RepetitionHistory& history, const SearchResult& search,
    chu::Move selected, int depth, MateGuardLimits& limits) {
  RootMateGuardResult result;
  result.move = selected;
  if (oddMateDepth(depth) == 0 || search.provenMate || search.allRootMovesLost)
    return result;
  auto order = search.rootMoves;
  std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
    return a.visits > b.visits;
  });
  if (order.empty()) return result;
  const bool flip = root.sideToMove == chu::Gote;
  auto path = history;
  if (path.empty()) path.start(root);
  // Include the existing immediate-capture backstop and visited-node proofs.
  for (auto& candidate : order) {
    if (std::find(search.unsafeRootMoves.begin(), search.unsafeRootMoves.end(),
                  moveToIndex(candidate.move)) != search.unsafeRootMoves.end()) {
      candidate.provenLoss = true;
      candidate.lossPlies = std::max(candidate.lossPlies, 2);
    }
  }
  while (true) {
    auto current = std::find_if(order.begin(), order.end(), [&](const auto& c) {
      return moveToIndex(c.move) == moveToIndex(result.move);
    });
    if (current == order.end()) return result;
    if (!current->provenLoss) {
      auto child = root;
      const auto actual = flip ? chu::flipMove(current->move) : current->move;
      path.append(root, actual);
      child.apply(actual);
      const auto proof = probeUsiMate(child, path, depth, limits);
      path.pop();
      if (proof.result != chu::MateResult::kMate) return result;
      current->provenLoss = true;
      current->lossPlies = 1 + proof.matePly;
      result.rejected.push_back(current->move);
    }
    auto next = std::find_if(order.begin(), order.end(), [](const auto& c) {
      return !c.provenLoss;
    });
    if (next != order.end()) {
      result.move = next->move;
      continue;
    }
    const auto slowest = std::max_element(order.begin(), order.end(),
        [](const auto& a, const auto& b) { return a.lossPlies < b.lossPlies; });
    result.move = slowest->move;
    result.allLost = true;
    result.lossPlies = slowest->lossPlies;
    return result;
  }
}

}  // namespace az
