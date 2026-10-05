#include "mcts.h"

#include <algorithm>
#include <cmath>

#include "../movegen.h"
#include "../status.h"

namespace az {

uint64_t positionKey(const chu::Position& pos) {
  uint64_t h = 1469598103934665603ull;
  for (int s = 0; s < chu::kSquares; ++s) {
    h ^= pos.board[s] + 1;
    h *= 1099511628211ull;
  }
  h ^= static_cast<uint64_t>(pos.sideToMove) + 1;
  h *= 1099511628211ull;
  h ^= static_cast<uint64_t>(pos.lastLionCapture + 2);
  h *= 1099511628211ull;
  return h;
}

namespace {

// Terminal value from the side-to-move perspective, or -1 if not terminal.
// The early in-tree repetition cutoff (any earlier occurrence = draw) follows
// JHBR3's convention.
int terminalOf(const chu::Position& pos, uint64_t hash,
               const std::vector<uint64_t>& gameHashes,
               const std::vector<uint64_t>& hashPath, int totalPly,
               int maxPlyCap) {
  chu::StatusResult st = chu::evaluateStatus(pos);
  if (st.end != chu::GameEnd::Playing) {
    if (st.winner < 0) return 2;  // draw
    return st.winner == pos.sideToMove ? 1 : 0;
  }
  if (totalPly >= maxPlyCap) return 2;  // ply cap
  for (uint64_t h : gameHashes)
    if (h == hash) return 2;
  for (uint64_t h : hashPath)
    if (h == hash) return 2;
  return -1;
}

}  // namespace

Mcts::Mcts(MctsConfig cfg) : cfg_(cfg) {}

int Mcts::newNode() {
  arena_.emplace_back();
  return static_cast<int>(arena_.size()) - 1;
}

void Mcts::fillNode(int nodeIdx, const chu::Position& pos,
                    const float* policy, const float* wdl) {
  Node& node = arena_[nodeIdx];
  node.leafValue = wdl[0] + wdl[1] * 0.5f;  // win + draw/2, stm perspective
  const std::vector<chu::Move> legal = chu::generateLegal(pos);
  std::vector<float> logits;
  logits.reserve(legal.size());
  float maxLogit = -1e30f;
  for (const chu::Move& m : legal) {
    const float l = policy[moveToIndex(m)];
    logits.push_back(l);
    maxLogit = std::max(maxLogit, l);
  }
  float sum = 0.f;
  for (float& l : logits) sum += (l = std::exp(l - maxLogit));
  node.edges.reserve(legal.size());
  for (size_t i = 0; i < legal.size(); ++i) {
    Edge e;
    e.move = legal[i];
    e.policyIndex = moveToIndex(legal[i]);
    e.prior = logits[i] / sum;
    node.edges.push_back(e);
  }
}

int Mcts::pickChild(int nodeIdx, bool isRoot) const {
  const Node& node = arena_[nodeIdx];
  const int parentN = std::max(1, node.visits);
  const float cNum = isRoot ? cfg_.cInitRoot : cfg_.cInit;
  const float cBase = isRoot ? cfg_.cBaseRoot : cfg_.cBase;
  const float c = cNum + std::log((parentN + cBase + 1) / cBase);
  const float sqrtN = std::sqrt(static_cast<float>(parentN));
  const float fpuRed = isRoot ? cfg_.fpuReductionRoot : cfg_.fpuReduction;

  float visitedMass = 0.f, sumW = 0.f;
  int32_t sumN = 0;
  for (const Edge& e : node.edges) {
    if (e.n > 0) visitedMass += e.prior;
    sumW += e.w;
    sumN += e.n;
  }
  const float parentQ = sumN > 0 ? sumW / sumN : 0.5f;
  const float fpu = parentQ - fpuRed * std::sqrt(visitedMass);

  int best = -1;
  float bestScore = -1e30f;
  for (size_t i = 0; i < node.edges.size(); ++i) {
    const Edge& e = node.edges[i];
    const float q = e.n + e.inflight > 0 ? e.w / (e.n + e.inflight) : fpu;
    const float u = c * e.prior * sqrtN / (1 + e.n + e.inflight);
    const float score = q + u;
    if (score > bestScore) {
      bestScore = score;
      best = static_cast<int>(i);
    }
  }
  return best;
}

// Descends from the root to an unevaluated/terminal leaf, applying virtual
// loss on the taken edges (undone when the batch is backed up).
Mcts::Descent Mcts::descend() {
  Descent d;
  d.pos = rootPos_;
  d.hashPath.clear();
  int nodeIdx = root_;
  while (true) {
    Node& node = arena_[nodeIdx];
    if (node.terminal >= 0) {  // terminal node: backup its value directly
      d.leaf = nodeIdx;
      d.needsEval = false;
      d.value = node.leafValue;
      return d;
    }
    if (!node.evaluated && nodeIdx != root_) {  // queued for eval already
      d.leaf = nodeIdx;
      d.needsEval = true;
      return d;
    }
    const int ei = pickChild(nodeIdx, nodeIdx == root_);
    Edge& e = node.edges[ei];
    e.inflight++;
    e.n++;
    node.visits++;
    d.path.emplace_back(nodeIdx, ei);
    d.pos.apply(e.move);
    const uint64_t h = positionKey(d.pos);
    if (e.child < 0) {
      e.child = newNode();
      Node& child = arena_[e.child];
      // repetition: against game history and earlier in-tree positions
      // (hashPath does not yet contain h)
      const int term = terminalOf(d.pos, h, gameHashes_, d.hashPath,
                                  gamePly_ + (int)d.hashPath.size(),
                                  cfg_.maxPlyCap);
      d.hashPath.push_back(h);
      if (term >= 0) {
        child.terminal = term;
        child.leafValue = term == 2 ? 0.5f : (term == 1 ? 1.0f : 0.0f);
        child.evaluated = true;
        d.leaf = e.child;
        d.needsEval = false;
        d.value = child.leafValue;
        return d;
      }
      d.leaf = e.child;
      d.needsEval = true;
      return d;
    }
    d.hashPath.push_back(h);
    nodeIdx = e.child;
  }
}

void Mcts::backup(const Descent& d) {
  // d.value is the leaf's value from the leaf's side-to-move perspective;
  // the last edge leads into the leaf, so it receives the flipped value.
  float v = 1.f - d.value;
  for (auto it = d.path.rbegin(); it != d.path.rend(); ++it) {
    Edge& e = arena_[it->first].edges[it->second];
    e.inflight--;  // the virtual visit becomes the real one
    e.w += v;
    v = 1.f - v;
  }
}

void Mcts::runBatch(Evaluator& eval) {
  std::vector<Descent> batch;
  batch.reserve(cfg_.batchSize);
  for (int i = 0; i < cfg_.batchSize; ++i) batch.push_back(descend());

  // Evaluate all non-terminal leaves in one call.
  int n = 0;
  for (const Descent& d : batch)
    if (d.needsEval) ++n;
  if (n > 0) {
    std::vector<float> planes(size_t(n) * kNumPlanes * kBoard);
    std::vector<float> policy(size_t(n) * kPolicySize);
    std::vector<float> wdl(size_t(n) * 3);
    int j = 0;
    for (const Descent& d : batch) {
      if (!d.needsEval) continue;
      const int repCount = static_cast<int>(std::count(
          gameHashes_.begin(), gameHashes_.end(), positionKey(d.pos)));
      encodePlanes(d.pos, repCount, gamePly_ + (int)d.hashPath.size(),
                   planes.data() + size_t(j) * kNumPlanes * kBoard);
      ++j;
    }
    eval.evaluate(n, planes.data(), policy.data(), wdl.data());
    j = 0;
    for (const Descent& d : batch) {
      if (!d.needsEval) continue;
      fillNode(d.leaf, d.pos, policy.data() + size_t(j) * kPolicySize,
               wdl.data() + size_t(j) * 3);
      ++j;
    }
  }
  for (const Descent& d : batch) backup(d);
}

SearchResult Mcts::search(const chu::Position& root,
                          const std::vector<uint64_t>& gameHashes, int gamePly,
                          Evaluator& eval, std::mt19937_64& rng) {
  arena_.clear();
  gameHashes_ = gameHashes;
  gamePly_ = gamePly;
  rootPos_ = root;

  SearchResult res;
  root_ = newNode();
  Node& rootNode = arena_[root_];
  // Root terminal check: engine status and the ply cap only — repetition at
  // the root is the game loop's call (in-tree early cutoff would otherwise
  // fire on the *second* occurrence, before the real fourfold draw).
  const chu::StatusResult st = chu::evaluateStatus(rootPos_);
  const bool rootTerminal = st.end != chu::GameEnd::Playing ||
                            gamePly_ >= cfg_.maxPlyCap;
  const int term = !rootTerminal ? -1
                   : st.winner < 0     ? 2
                   : st.end == chu::GameEnd::Playing ? 2
                   : st.winner == rootPos_.sideToMove ? 1
                                                      : 0;
  if (term >= 0) {
    rootNode.terminal = term;
    rootNode.leafValue = term == 2 ? 0.5f : (term == 1 ? 1.0f : 0.0f);
    return res;
  }
  {
    // root evaluation (single-position batch)
    std::vector<float> planes(kNumPlanes * kBoard), policy(kPolicySize),
        wdl(3);
    const int repCount = static_cast<int>(std::count(
        gameHashes_.begin(), gameHashes_.end(), positionKey(rootPos_)));
    encodePlanes(rootPos_, repCount, gamePly_, planes.data());
    eval.evaluate(1, planes.data(), policy.data(), wdl.data());
    fillNode(root_, rootPos_, policy.data(), wdl.data());
  }
  rootNode.evaluated = true;
  if (rootNode.edges.empty()) return res;

  if (cfg_.rootNoise) {
    std::gamma_distribution<float> gamma(cfg_.dirichletAlpha, 1.0f);
    std::vector<float> noise(rootNode.edges.size());
    float sum = 0.f;
    for (float& x : noise) sum += (x = gamma(rng));
    for (size_t i = 0; i < noise.size(); ++i)
      rootNode.edges[i].prior =
          (1 - cfg_.dirichletEps) * rootNode.edges[i].prior +
          cfg_.dirichletEps * noise[i] / sum;
  }

  for (int done = 0; done < cfg_.sims; done += cfg_.batchSize) runBatch(eval);

  float sumW = 0.f;
  int32_t sumN = 0;
  for (const Edge& e : rootNode.edges) {
    sumW += e.w;
    sumN += e.n;
  }
  res.rootVisits = sumN;
  res.rootQ = sumN > 0 ? sumW / sumN : rootNode.leafValue;
  res.policy.reserve(rootNode.edges.size());
  const Edge* best = nullptr;
  for (const Edge& e : rootNode.edges) {
    res.policy.emplace_back(e.policyIndex, sumN ? float(e.n) / sumN : 0.f);
    if (!best || e.n > best->n) best = &e;
  }
  if (best) res.bestMove = best->move;

  // A0GB target: greedy max-visit descent; the target is the terminal value
  // at a terminal node, else the leaf net eval, negated back to the root's
  // perspective per descended ply.
  {
    int nodeIdx = root_;
    int depth = 0;
    float v = rootNode.leafValue;
    while (true) {
      const Node& n = arena_[nodeIdx];
      if (n.terminal >= 0) {
        v = n.leafValue;
        break;
      }
      const Edge* be = nullptr;
      for (const Edge& e : n.edges)
        if (e.child >= 0 && (!be || e.n > be->n)) be = &e;
      if (!be) break;                     // this node was never descended into
      if (arena_[be->child].visits <= 1)  // freshly evaluated leaf: stop
        break;
      nodeIdx = be->child;
      v = arena_[nodeIdx].leafValue;
      ++depth;
    }
    res.a0gb = (depth % 2) ? 1.f - v : v;
  }
  return res;
}

}  // namespace az
