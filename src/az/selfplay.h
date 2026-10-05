// AlphaZero self-play game loop and record writing.
//
// A game is played from an initial SFEN; each ply runs MCTS on the
// canonicalized position, records the per-position training sample (packed
// planes + visit policy + the three value targets), and plays a move
// (temperature-sampled for the first plies, argmax after). Terminal
// adjudication reuses the engine's status rules plus fourfold repetition
// and the 1000-ply cap, exactly like tools/random_match.py.
//
// Game records are JSON (initial SFEN + USI moves + samples), readable by
// the Python replay buffer and validatable with the existing engine.

#pragma once

#include <string>
#include <vector>

#include "../position.h"
#include "mcts.h"

namespace az {

struct Sample {
  int ply = 0;
  int stm = 0;  // 0 = sente, 1 = gote (original frame)
  std::vector<uint8_t> packedPlanes;  // 81 bit planes (progress separate)
  float progress = 0.f;               // ply / 1000
  std::vector<std::pair<int32_t, float>> policy;
  float rootQ = 0.5f;
  float a0gb = 0.5f;
  float z = 0.5f;  // game result from stm's perspective (filled at game end)
};

struct GameRecord {
  std::string initialSfen;
  std::vector<std::string> moves;  // USI, original frame
  std::vector<Sample> samples;
  int result = 0;  // 1 = sente wins, -1 = gote wins, 0 = draw
  std::string reason;
};

struct SelfplayConfig {
  int sims = 200;
  int batchSize = 16;
  int tempPlies = 20;  // sample with temperature for this many plies
  uint64_t seed = 1;
  MctsConfig mcts;
};

GameRecord playGame(const std::string& initialSfen, const SelfplayConfig& cfg,
                    Evaluator& eval);

void writeGameRecord(const GameRecord& rec, const std::string& path);

// Uniform-policy / constant-value evaluator (M2 smoke test).
struct RandomEval : Evaluator {
  void evaluate(int n, const float*, float* policyOut, float* wdlOut) override;
};

}  // namespace az
