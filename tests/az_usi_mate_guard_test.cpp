#include <iostream>
#include <stdexcept>
#include "../src/az/usi_mate_guard.h"
#include "../src/az/selfplay.h"
#include "../src/sfen.h"
#include "../src/usi.h"

void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
chu::Position position(const char* text) {
  chu::Position p;
  check(chu::parseSfen(text, p), "parse SFEN");
  return p;
}
chu::Move move(const char* text) {
  chu::Move m;
  check(chu::parseUsiMove(text, m), "parse move");
  return m;
}
chu::MateAnswer probe(const chu::Position& p, int depth, size_t nodes = 200000) {
  az::MateGuardLimits limits;
  limits.nodeLimit = nodes;
  auto start = std::chrono::steady_clock::now();
  auto result = az::probeUsiMate(p, {}, depth, limits);
  std::cout << "depth " << depth << " result " << int(result.result)
            << " plies " << result.matePly << " nodes " << result.nodes
            << " ms " << std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now() - start).count() << '\n';
  return result;
}

int main() {
  using Result = chu::MateResult;
  check(az::oddMateDepth(-1) == 0 && az::oddMateDepth(0) == 0 &&
        az::oddMateDepth(2) == 1 && az::oddMateDepth(6) == 5 &&
        az::oddMateDepth(7) == 7 && az::oddMateDepth(8) == 7, "odd depth rounding");
  auto p = position("5k6/12/12/12/12/12/12/12/10G1/g2r8/1K10/12 b - 1");
  auto losing = move("11k12l");
  check(!az::hangsLastRoyal(p, losing), "fixture requires a mating check, not a hanging king");
  auto child = p;
  child.apply(losing);
  check(probe(child, 7).result == Result::kMate, "seven-ply search proves fixture");
  check(probe(child, 1).result == Result::kMate, "one-ply mating check boundary");
  check(probe(child, 0).result == Result::kUnknown, "zero disables");
  auto mate3 = position("10k1/11+G/9+E1K/8n3/11L/12/12/12/12/12/12/12 b - 1");
  check(probe(mate3, 1).result != Result::kMate, "three-ply fixture is not mate in one");
  check(probe(mate3, 3).result == Result::kMate, "three-ply horizon includes final check");
  auto mate5 = position("12/12/11k/1R10/R11/12/12/12/12/12/12/g5K5 b - 1");
  check(probe(mate5, 3).result != Result::kMate, "five-ply fixture is not mate in three");
  check(probe(mate5, 5).result == Result::kMate, "five-ply horizon includes final check");
  auto mate7 = position("12/12/12/11k/1R10/R11/12/12/12/12/12/g5K5 b - 1");
  check(probe(mate7, 5).result != Result::kMate, "seven-ply fixture is not mate in five");
  check(probe(mate7, 7).result == Result::kMate, "seven-ply horizon proves the ladder");
  // LeafMateDepth maps the conventional horizon to the BNS boundary ply.
  chu::MateSolver leaf(1);
  leaf.setMaxPly(4);
  check(leaf.solve(mate5, 20000).result != Result::kMate, "leaf depth three excludes mate five");
  leaf.setMaxPly(6);
  check(leaf.solve(mate5, 20000).result == Result::kMate, "leaf depth five includes final AND node");
  check(probe(child, 7, 1).result == Result::kUnknown, "node exhaustion is unknown");
  auto capped = child;
  capped.moveNumber = 1000;
  check(probe(capped, 7).result != Result::kMate, "game cap before royal capture prevents proof");
  capped.moveNumber = 999;
  check(probe(capped, 1).result != Result::kMate, "a legal non-evasion can reach the cap before capture");
  capped.moveNumber = 998;
  check(probe(capped, 1).result == Result::kMate, "royal capture on the final allowed ply wins");
  auto prince = child;
  prince.board[chu::makeSq(0, 10)] = chu::makePiece(chu::Prince, chu::Sente);
  check(probe(prince, 7).result == Result::kUnknown, "multi-royal defender remains unknown");
  auto countercapture = child;
  countercapture.board[chu::makeSq(6, 2)] = chu::makePiece(chu::Rook, chu::Sente);
  check(probe(countercapture, 1).result != Result::kMate,
        "defender can capture the attacker's last royal instead of evading");
  auto promotionEscape = child;
  promotionEscape.board[chu::makeSq(0, 4)] = chu::makePiece(chu::Elephant, chu::Sente);
  check(probe(promotionEscape, 1).result != Result::kMate,
        "defender can create a prince while ignoring the check");
  std::atomic<bool> stop{true};
  az::MateGuardLimits stopped;
  stopped.stop = &stop;
  check(az::probeUsiMate(child, {}, 7, stopped).result == Result::kUnknown &&
        stopped.nodes == 0 && stopped.cancelled, "stop cancels immediately");
  az::MateGuardLimits expired;
  expired.deadline = std::chrono::steady_clock::now();
  check(az::probeUsiMate(child, {}, 7, expired).result == Result::kUnknown &&
        expired.nodes == 0 && expired.cancelled, "deadline cancels immediately");

  auto repeating = position("k11/g11/12/12/12/12/12/12/12/12/11G/11K b - 1");
  az::RepetitionHistory history;
  history.start(repeating);
  const char* cycle[] = {"1l2l", "12a11a", "2l1l", "11a12a"};
  for (int i = 0; i < 12; ++i) {
    auto m = move(cycle[i % 4]);
    history.append(repeating, m);
    repeating.apply(m);
  }
  az::MateGuardLimits repeated;
  check(history.verdict().fourfold && history.verdict().loser < 0,
        "fixture is an actual history-dependent draw");
  check(az::probeUsiMate(repeating, history, 7, repeated).result == Result::kNoMate &&
        repeated.nodes == 4, "each iterative depth stops at the already drawn root");
  auto checking = position("6k1s3/6e5/8c3/5+O4+M1/2p9/12/12/2P9/F1M7T1/1S8S1/6E5/6K5 b - 447");
  history.start(checking);
  const char* checks[] = {"7d8c", "6a5b", "8c7d", "5b6a"};
  for (int i = 0; i < 12; ++i) {
    auto m = move(checks[i % 4]);
    history.append(checking, m);
    checking.apply(m);
  }
  az::MateGuardLimits foul;
  check(history.verdict().loser == chu::Sente &&
        az::probeUsiMate(checking, history, 7, foul).result == Result::kNoMate,
        "perpetual attack loss cannot be called an attacker mate");

  az::SearchResult search;
  search.bestMove = move("11k10k");  // sampled selection differs from greedy
  search.rootMoves.push_back({losing, 100});
  search.rootMoves.push_back({move("11k10k"), 50});
  az::MateGuardLimits limits;
  auto guarded = az::guardUsiMove(p, {}, search, losing, 7, limits);
  check(!guarded.rejected.empty() &&
        az::moveToIndex(guarded.move) != az::moveToIndex(losing), "reject selected mating loss");
  auto white = p.flipped();
  az::MateGuardLimits mirrorLimits;
  auto mirrored = az::guardUsiMove(white, {}, search, losing, 7, mirrorLimits);
  check(az::moveToIndex(mirrored.move) == az::moveToIndex(guarded.move), "both colors use canonical moves");
  az::MateGuardLimits noNodes;
  noNodes.nodeLimit = 0;
  check(az::moveToIndex(az::guardUsiMove(p, {}, search, losing, 7, noNodes).move)
        == az::moveToIndex(losing), "unknown keeps candidate");
  search.provenMate = true;
  az::MateGuardLimits won;
  az::guardUsiMove(p, {}, search, losing, 7, won);
  check(won.nodes == 0, "offensive mate override keeps priority");
  search.provenMate = false;
  search.rootMoves[1].provenLoss = true;
  search.rootMoves[1].lossPlies = 20;
  az::MateGuardLimits allLimits;
  auto all = az::guardUsiMove(p, {}, search, losing, 7, allLimits);
  check(all.allLost && all.lossPlies == 20 &&
        az::moveToIndex(all.move) == az::moveToIndex(search.rootMoves[1].move),
        "all-lost fallback chooses longest available proof line");
  std::cout << "USI mate guard checks passed\n";
}
