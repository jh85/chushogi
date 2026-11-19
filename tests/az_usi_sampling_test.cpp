#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../src/az/selfplay.h"
#include "../src/az/usi_sampling.h"
#include "../src/sfen.h"
#include "../src/usi.h"

void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
chu::Move move(const char* text) {
  chu::Move m;
  check(chu::parseUsiMove(text, m), "parse move");
  return m;
}
bool same(chu::Move a, chu::Move b) {
  return chu::moveToUsi(a) == chu::moveToUsi(b);
}

int main() {
  const auto a = move("12i12h"), b = move("11i11h"), c = move("10i10h");
  const int ai = az::moveToIndex(a), bi = az::moveToIndex(b);
  az::SearchResult result;
  result.bestMove = a;
  result.policy = {{ai, .75f}, {bi, .25f}, {az::moveToIndex(c), 0.f}};
  const auto original = result.policy;
  for (double t : {0.5, 1.0, 2.0}) {
    std::mt19937_64 rng(123);
    int alternatives = 0;
    for (int i = 0; i < 30000; ++i) {
      const auto chosen = az::selectUsiMove(result, 16, 15, t, rng);
      check(!same(chosen, c), "zero-visit moves must not be sampled");
      alternatives += same(chosen, b);
    }
    const double expected = 1 / (1 + std::pow(3., 1 / t));
    check(std::abs(alternatives / 30000. - expected) < .015,
          "visit temperature must give N^(1/T) probabilities");
  }
  check(result.policy == original, "sampling must not modify visits");
  for (auto cutoff : {std::pair<int,int>{0, 0}, {16, 16}, {16, 999}}) {
    std::mt19937_64 rng(42), copy = rng;
    check(same(az::selectUsiMove(result, cutoff.first, cutoff.second, .5, rng), a),
          "cutoff must return the original safe greedy pick");
    check(rng == copy, "greedy selection must not consume randomness");
  }
  std::mt19937_64 rng(123), copy = rng;
  result.bestMove = c;  // zero visits but selected by the safety fallback
  result.provenMate = true;
  check(same(az::selectUsiMove(result, 16, 0, .5, rng), c), "preserve proved mate");
  result.provenMate = false;
  result.allRootMovesLost = true;
  check(same(az::selectUsiMove(result, 16, 0, .5, rng), c), "preserve slowest loss");
  result.allRootMovesLost = false;
  result.unsafeRootMoves = {ai};
  check(same(az::selectUsiMove(result, 16, 0, .5, rng), b), "exclude unsafe visits");
  result.unsafeRootMoves.push_back(bi);
  check(same(az::selectUsiMove(result, 16, 0, .5, rng), c),
        "if safe visit mass is zero retain safe bestMove");
  check(rng == copy, "forced and fallback picks must not consume randomness");
  result.unsafeRootMoves.clear(); result.bestMove = a;
  check(same(az::selectUsiMove(result, 16, 0, 1e-300, rng), a),
        "very small temperatures must remain numerically stable");

  // Real proved losses: 4j5i has all the visits but hangs the last royal.
  struct ForcedEval : az::Evaluator {
    void evaluate(int n, const float*, float* p, float* v) override {
      std::fill(p, p + size_t(n) * az::kPolicySize, -1000.f);
      for (int i = 0; i < n; ++i) {
        p[size_t(i) * az::kPolicySize + az::moveToIndex(move("4j5i"))] = 1000.f;
        v[3*i] = v[3*i+1] = v[3*i+2] = 0.f;
      }
    }
  } prior;
  chu::Position position;
  check(chu::parseSfen("4gtkt4/4se1h3f/f9m1/1c8o1/12/10s1/12/12/10+c1/2S5K3/12/2C5+r3 b - 319", position), "parse safety fixture");
  az::MctsConfig cfg;
  cfg.sims = cfg.batchSize = 16; cfg.rootNoise = false;
  cfg.rootProofSelection = cfg.rootRoyalFilter = cfg.safeSampling = true;
  cfg.probeDepth = 3;
  auto safe = az::Mcts(cfg).search(position, {}, 318, prior, rng);
  check(!safe.unsafeRootMoves.empty(), "MCTS must provide sampling exclusions");
  for (int i = 0; i < 1000; ++i)
    check(!az::hangsLastRoyal(position, az::selectUsiMove(safe, 1000, 318, .5, rng)),
          "sampling must preserve real proof and royal safety, even without safe visits");
  std::cout << "USI visit temperature, cutoff, forced mate, slowest loss and safety checks passed\n";
}
