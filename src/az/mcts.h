// AlphaZero-style MCTS (PUCT) for chu shogi.
//
// dlshogi/JHBR3 conventions: c_init + log-scaled c_base, FPU for unvisited
// edges (parent Q minus a reduction over the visited policy mass), virtual
// loss for batched leaf collection, root Dirichlet noise for self-play.
//
// Values are win probabilities in [0, 1] from the perspective of the side to
// move at each node. Edge statistics (n, w) live in the parent's perspective.
//
// Per root search, search() returns the visit-count policy plus the three
// value targets used by the trainer: z (filled in by the game loop), rootQ
// (soft-Z: the root's mean backed-up value) and the A0GB target (value at the
// end of the greedy max-visit path — the leaf's net eval, or the true outcome
// when the path reaches a terminal).

#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <random>
#include <vector>

#include "../position.h"
#include "encode.h"

namespace az {

struct Evaluator {
  virtual ~Evaluator() = default;
  // Batch-evaluate canonical (side-to-move frame) positions.
  // planesIn: n * kNumPlanes * kBoard floats.
  // policyOut: n * kPolicySize logits; wdlOut: n * 3 floats.
  virtual void evaluate(int n, const float* planesIn, float* policyOut,
                        float* wdlOut) = 0;
};

struct MctsConfig {
  int sims = 200;
  int batchSize = 16;
  float cInit = 1.25f;
  float cBase = 19652.f;
  float cInitRoot = 1.25f;
  float cBaseRoot = 19652.f;
  float fpuReduction = 0.27f;
  float fpuReductionRoot = 0.0f;
  float dirichletEps = 0.25f;
  float dirichletAlpha = 0.15f;
  bool rootNoise = true;  // self-play on; eval matches off
  int maxPlyCap = 1000;   // game draw cap (plies from game start)
};

struct Edge {
  chu::Move move;
  int32_t policyIndex = -1;
  int32_t child = -1;  // arena index; -1 until first expansion
  float prior = 0.f;
  int32_t n = 0;
  float w = 0.f;      // backed-up value sum, parent (mover) perspective
  int32_t inflight = 0;
};

struct Node {
  std::vector<Edge> edges;
  int8_t terminal = -1;    // -1 none; 0 stm loses; 1 stm wins; 2 draw
  bool evaluated = false;  // leafValue/edges filled
  float leafValue = 0.5f;  // net eval (or terminal value), stm perspective
  int32_t visits = 0;      // descents through this node
};

struct SearchResult {
  chu::Move bestMove;
  std::vector<std::pair<int, float>> policy;  // policy index, visit prob
  float rootQ = 0.5f;
  float a0gb = 0.5f;
  int32_t rootVisits = 0;
};

class Mcts {
 public:
  explicit Mcts(MctsConfig cfg = {});

  // Runs cfg.sims playouts from `root` (canonical frame). `gameHashes` =
  // 64-bit keys of all earlier game positions (repetition detection);
  // `gamePly` = plies already played (for the 1000-ply cap).
  SearchResult search(const chu::Position& root,
                      const std::vector<uint64_t>& gameHashes, int gamePly,
                      Evaluator& eval, std::mt19937_64& rng);

 private:
  struct Descent {
    std::vector<std::pair<int, int>> path;  // (node, edge) from root
    chu::Position pos;                      // position at the leaf
    std::vector<uint64_t> hashPath;         // in-tree position keys
    int leaf = -1;                          // arena index of the leaf
    bool needsEval = false;
    float value = 0.5f;                     // terminal value if !needsEval
  };

  int newNode();
  void fillNode(int nodeIdx, const chu::Position& pos, const float* policy,
                const float* wdl);
  Descent descend();
  void backup(const Descent& d);
  void runBatch(Evaluator& eval);
  int pickChild(int nodeIdx, bool isRoot) const;

  MctsConfig cfg_;
  std::deque<Node> arena_;
  int root_ = -1;
  chu::Position rootPos_;
  std::vector<uint64_t> gameHashes_;
  int gamePly_ = 0;
};

// Stable 64-bit key of a position (board + side + lion history) for
// repetition detection.
uint64_t positionKey(const chu::Position& pos);

}  // namespace az
