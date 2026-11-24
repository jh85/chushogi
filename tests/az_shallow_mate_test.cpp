#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include "../src/az/shallow_mate.h"
#include "../src/az/selfplay.h"
#include "../src/attacks.h"
#include "../src/movegen.h"
#include "../src/sfen.h"
#include "../src/status.h"
#include "../src/usi.h"

void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
chu::Position position(const char* text) {
  chu::Position p;
  check(chu::parseSfen(text, p), "SFEN parse");
  return p;
}
std::vector<std::string> tokens(const std::vector<chu::Move>& moves) {
  std::vector<std::string> out;
  for (const auto& move : moves) out.push_back(chu::moveToUsi(move));
  std::sort(out.begin(), out.end());
  return out;
}

int main() {
  using Result = chu::MateResult;
  std::vector<chu::Position> positions;
  for (const char* text : {
      "lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1",
      "4gtkt4/4se1h3f/f9m1/1c8o1/12/10s1/12/12/10+c1/2S5K3/12/2C5+r3 b - 319",
      "12/7Q4/7N3n/10g1/10P1/9P1k/10+p1/9qt1/12/12/12/K11 b - 1",
      "10k1/11+G/9+E1K/8n3/11L/12/12/12/12/12/12/12 b - 1",
      "12/12/12/11k/1R10/R11/12/12/12/12/12/g5K5 b - 1"})
    positions.push_back(position(text));
  std::mt19937_64 rng(42);
  for (int sample = 0; sample < 128; ++sample) {
    chu::Position p;
    for (auto color : {chu::Sente, chu::Gote}) {
      int square;
      do { square = rng() % chu::kSquares; } while (p.board[square]);
      p.board[square] = chu::makePiece(chu::King, color);
      for (int i = 0; i < 10; ++i) {
        do { square = rng() % chu::kSquares; } while (p.board[square]);
        chu::Role role;
        do { role = static_cast<chu::Role>(rng() % chu::RoleCount); }
        while (role == chu::King || role == chu::Prince);
        p.board[square] = chu::makePiece(role, color);
        if (chu::isLion(role)) p.lastLionCapture = square;
      }
    }
    positions.push_back(p);
  }
  auto game = positions.front();
  for (int ply = 0; ply < 150; ++ply) {
    auto legal = chu::generateLegal(game);
    if (legal.empty() || chu::evaluateStatus(game).end != chu::GameEnd::Playing) break;
    game.apply(legal[rng() % legal.size()]);
    if (ply % 5 == 0) positions.push_back(game);
  }
  size_t checkedMoves = 0;
  for (const auto& original : positions) for (bool flip : {false, true})
    for (auto rules : {chu::RuleSet::Lishogi, chu::RuleSet::JCSA}) {
      auto p = flip ? original.flipped() : original;
      p.rules = rules;
      const auto legal = chu::generateLegal(p);
      check(az::canonicalPositionKey(p) == az::positionKey(
                p.sideToMove == chu::Gote ? p.flipped() : p),
            "direct canonical hashing matches board reflection, including lion history");
      int royal = chu::kNoSquare, count = 0;
      for (int sq = 0; sq < chu::kSquares; ++sq) {
        const auto piece = p.board[sq];
        if (piece && chu::colorOf(piece) != p.sideToMove && chu::isRoyal(chu::roleOf(piece))) {
          royal = sq;
          ++count;
        }
      }
      std::vector<chu::Move> checks, captures;
      for (const auto& move : legal) {
        const chu::MoveUndo undo(p, move);
        const auto before = p;
        p.apply(move);
        if (count == 1) {
          const bool captured = move.to == royal || move.mid == royal;
          if (captured) captures.push_back(move);
          if (captured || chu::inCheck(p, p.sideToMove)) checks.push_back(move);
        }
        undo.restore(p);
        check(p.board == before.board && p.sideToMove == before.sideToMove &&
              p.lastLionCapture == before.lastLionCapture && p.moveNumber == before.moveNumber,
              "undo must restore every move, promotion and lion return");
        ++checkedMoves;
      }
      check(chu::evaluateStatus(p).end == chu::evaluateStatus(p, !legal.empty()).end,
            "status with cached move availability");
      if (count == 1) {
        check(tokens(checks) == tokens(chu::generateChecking(p, royal)),
              "checking generator matches exhaustive legality + check filter");
        check(tokens(captures) == tokens(chu::generateRoyalCaptures(p, royal)),
              "royal-capture generator matches exhaustive legal captures");
      }
    }

  auto root = position("5k6/12/12/12/12/12/12/12/10G1/g2r8/1K10/12 b - 1");
  chu::Move losing;
  check(chu::parseUsiMove("11k12l", losing), "parse losing move");
  auto child = root;
  child.apply(losing);
  const auto key = az::positionKey(child.flipped());
  az::ShallowMateLimits emptyLimits;
  std::vector<uint64_t> previous;
  check(az::probeShallowMate(child, previous, 3, emptyLimits).result == Result::kMate,
        "leaf mode proves non-immediate mate");
  az::ShallowMateLimits repeatedLimits;
  previous.push_back(key);
  check(az::probeShallowMate(child, previous, 3, repeatedLimits).result == Result::kUnknown,
        "key-only leaf history never proves a mate through repetition");
  std::unordered_set<uint64_t> gameKeys{key};
  az::ShallowMateHistory cached{&gameKeys, &previous, 0};
  az::ShallowMateLimits cachedLimits;
  check(az::probeShallowMate(child, cached, 3, cachedLimits).result == Result::kUnknown,
        "cached game history also blocks repetition proofs");
  gameKeys.clear();
  az::ShallowMateLimits currentExcluded;
  check(az::probeShallowMate(child, cached, 3, currentExcluded).result == Result::kMate,
        "current leaf is excluded from the ancestor view");
  for (const char* tactical : {
      "12/7Q4/7N3n/10g1/10P1/9P1k/10+p1/9qt1/12/12/12/K11 b - 1",
      "6r1g3/6s1m3/5+P1k1ts1/6p+M1p2/8G3/5+H+r3N1/11+a/12/12/12/12/K11 b - 1"}) {
    auto puzzle = position(tactical);
    az::ShallowMateLimits leafLimits;
    const std::vector<uint64_t> noHistory;
    check(az::probeShallowMate(puzzle, noHistory, 5, leafLimits).result == Result::kMate,
          "default 20k shallow budget retains tactical five-ply proof");
    chu::MateSolver bns(1);
    bns.setMaxPly(6);
    check(bns.solve(puzzle, 2000).result == Result::kMate,
          "tactical coverage reference uses the legacy 2k BNS budget");
  }

  struct Prior : az::Evaluator {
    int policy;
    explicit Prior(chu::Move m) : policy(az::moveToIndex(m)) {}
    void evaluate(int n, const float*, float* p, float* v) override {
      std::fill(p, p + size_t(n) * az::kPolicySize, -1000.f);
      std::fill(v, v + 3 * n, 0.f);
      for (int i = 0; i < n; ++i) p[size_t(i) * az::kPolicySize + policy] = 1000.f;
    }
  } eval(losing);
  az::MctsConfig cfg;
  cfg.sims = cfg.batchSize = 16;
  cfg.rootNoise = false;
  cfg.collectStats = true;
  cfg.rootProofSelection = true;
  cfg.shallowMateDepth = 3;
  auto result = az::Mcts(cfg).search(root, {}, 0, eval, rng);
  check(chu::moveToUsi(result.bestMove) != "11k12l", "MCTS consumes shared leaf proof");
  check(std::any_of(result.rootMoves.begin(), result.rootMoves.end(), [&](const auto& row) {
    return az::moveToIndex(row.move) == az::moveToIndex(losing) && row.provenLoss;
  }), "new leaf routine marks proved loss independent of prior");
  std::cout << "shallow mate: " << checkedMoves
            << " undo/check-generation cases, rule sets, history and MCTS integration passed\n";
}
