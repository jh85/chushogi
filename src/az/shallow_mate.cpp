#include "shallow_mate.h"

#include <array>
#include "../attacks.h"
#include "../movegen.h"
#include "../status.h"
#include "mcts.h"

namespace az {

bool ShallowMateLimits::poll() {
  if (cancelled || nodes >= nodeLimit ||
      (stop && stop->load(std::memory_order_relaxed)) ||
      std::chrono::steady_clock::now() >= deadline) {
    cancelled = true;
    return false;
  }
  return true;
}

namespace {
using Result = chu::MateResult;
struct Proof { Result result = Result::kUnknown; int plies = 0; };

struct Material {
  int royals[2]{};
  int live[2]{};
  int defenderSquare = chu::kNoSquare;

  void add(uint8_t piece, int square, int delta, chu::Color defender) {
    if (!piece) return;
    const auto color = chu::colorOf(piece);
    const auto role = chu::roleOf(piece);
    if (chu::isRoyal(role)) {
      royals[color] += delta;
      if (color == defender && delta > 0) defenderSquare = square;
    }
    if (!((role == chu::Pawn || role == chu::Lance) &&
          chu::sqY(square) == chu::backrankY(color))) live[color] += delta;
  }
};

template <bool FullHistory>
class Search {
 public:
  Search(const chu::Position& root, const RepetitionHistory* history,
         const ShallowMateHistory* previous, ShallowMateLimits& limits, int cap)
      : root_(root), pos_(root), attacker_(root.sideToMove), history_(history),
        previous_(previous), limits_(limits), cap_(cap) {
    for (int sq = 0; sq < chu::kSquares; ++sq)
      material_.add(pos_.board[sq], sq, 1, !attacker_);
    route_.reserve(10);
  }

  Proof solve(int depth) {
    switch (depth) {
      case 1: return attack<1>();
      case 3: return attack<3>();
      case 5: return attack<5>();
      case 7: return attack<7>();
      default: return {};
    }
  }

 private:
  // No board copy or allocation per move; restore captures, promotion and
  // lion-trade history exactly, including igui and double captures.
  struct Played {
    Search& search;
    chu::MoveUndo undo;
    Material old;
    Played(Search& s, const chu::Move& move) : search(s), undo(s.pos_, move), old(s.material_) {
      for (int i = 0; i < undo.count; ++i)
        s.material_.add(undo.pieces[i], undo.squares[i], -1, !s.attacker_);
      s.pos_.apply(move);
      for (int i = 0; i < undo.count; ++i)
        s.material_.add(s.pos_.board[undo.squares[i]], undo.squares[i], 1, !s.attacker_);
      s.route_.push_back(move);
    }
    ~Played() {
      undo.restore(search.pos_);
      search.material_ = old;
      search.route_.pop_back();
    }
  };

  bool visit() {
    if (!limits_.poll()) return false;
    ++limits_.nodes;
    return true;
  }

  bool repetition(Proof& out) {
    const auto key = FullHistory ? positionKey(pos_) : canonicalPositionKey(pos_);
    const size_t ply = route_.size();
    keys_[ply] = key;
    if constexpr (FullHistory) {
      int count = history_->occurrences(key);
      for (size_t i = 1; i <= ply; ++i) count += keys_[i] == key;
      if (count < 4) return false;
      // Attack/chase counters are needed only on an actual fourth board
      // occurrence. Replay lazily instead of generating threat moves on
      // every ordinary branch of the checking tree.
      auto history = *history_;
      auto board = root_;
      for (const auto& move : route_) {
        history.append(board, move);
        board.apply(move);
      }
      const auto verdict = history.verdict();
      if (!verdict.fourfold) return false;
      out = {verdict.loser == !attacker_ ? Result::kMate : Result::kNoMate, 0};
    } else {
      const bool inGame = previous_->game && previous_->game->find(key) != previous_->game->end();
      const bool inPath = previous_->path &&
          std::find(previous_->path->begin(), previous_->path->begin() + previous_->pathLength, key)
              != previous_->path->begin() + previous_->pathLength;
      if (!inGame && !inPath &&
          std::find(keys_.begin(), keys_.begin() + ply, key) == keys_.begin() + ply)
        return false;
      out = {};
    }
    return true;
  }

  bool terminal(Proof& out, const bool* hasLegal = nullptr) {
    if (material_.royals[pos_.sideToMove] == 0) {
      out = {pos_.sideToMove != attacker_ ? Result::kMate : Result::kNoMate, 0};
      return true;
    }
    // With more than one live piece on both sides, neither bare king nor
    // insufficient material can apply. Counts are updated with the move.
    const bool materialSafe = material_.live[0] > 1 && material_.live[1] > 1;
    if (!materialSafe || (hasLegal && !*hasLegal)) {
      const auto status = chu::evaluateStatus(pos_, hasLegal ? *hasLegal : true);
      if (status.end != chu::GameEnd::Playing) {
        out = {status.winner == attacker_ ? Result::kMate : Result::kNoMate, 0};
        return true;
      }
    }
    if (repetition(out)) {
      // At an OR node an actual stalemate precedes a repetition award.
      if (out.result == Result::kMate && !hasLegal && chu::generateLegal(pos_).empty())
        out = {Result::kNoMate, 0};
      return true;
    }
    if (pos_.moveNumber - 1 >= cap_) {
      out = {Result::kNoMate, 0};
      return true;
    }
    if (material_.royals[!attacker_] != 1) { out = {}; return true; }
    return false;
  }

  bool startAttack(Proof& out) {
    if (!visit()) { out = {}; return true; }
    if (terminal(out)) return true;
    // Most evasions leave the royal unattacked: avoid generating capture
    // moves at all in that common case. The legal generator still checks
    // lion-trade rules when an actual capture is possible.
    const int royal = material_.defenderSquare;
    if (chu::threatened(pos_, attacker_, royal, false, chu::kNoSquare, chu::kNoSquare)) {
      const auto captures = chu::generateRoyalCaptures(pos_, royal);
      if (!captures.empty()) {
        if (!visit()) out = {};
        else out = {Result::kMate, 1};
        return true;
      }
    }
    return false;
  }

  template <int Depth>
  Proof defend() {
    if (!visit()) return {};
    // A final royal capture is already terminal; don't generate replies.
    if (material_.royals[pos_.sideToMove] == 0) return {Result::kMate, 0};
    auto replies = chu::generateLegal(pos_);
    const bool hasLegal = !replies.empty();
    Proof out;
    if (terminal(out, &hasLegal)) return out;
    // Try direct escapes first. Non-evasions must still be checked under
    // chu rules, but most are refuted by one immediate royal capture.
    std::stable_partition(replies.begin(), replies.end(), [&](const auto& move) {
      if (chu::isRoyal(chu::roleOf(pos_.board[move.from]))) return true;
      if (move.promote && chu::promoteRole(chu::roleOf(pos_.board[move.from])) == chu::Prince)
        return true;
      for (int sq : {int(move.to), int(move.mid)}) {
        if (sq == chu::kNoSquare) continue;
        const auto piece = pos_.board[sq];
        if (piece && chu::colorOf(piece) == attacker_ && chu::isRoyal(chu::roleOf(piece)))
          return true;
      }
      return false;
    });
    bool unknown = false;
    int longest = 0;
    for (const auto& reply : replies) {
      if (!limits_.poll()) return {};
      Played played(*this, reply);
      const auto child = attack<Depth - 1>();
      if (limits_.cancelled) return {};
      if (child.result == Result::kNoMate) return {Result::kNoMate, 0};
      unknown |= child.result == Result::kUnknown;
      longest = std::max(longest, 1 + child.plies);
    }
    return unknown ? Proof{} : Proof{Result::kMate, longest};
  }

  std::vector<chu::Move> checks() const {
    auto moves = chu::generateChecking(pos_, material_.defenderSquare);
    auto priority = [&](const chu::Move& move) {
      int score = std::max(0, 4 - chu::dist(move.to, material_.defenderSquare)) * 4;
      for (int sq : {int(move.to), int(move.mid)}) {
        if (sq != chu::kNoSquare && pos_.board[sq] &&
            chu::colorOf(pos_.board[sq]) != attacker_) score += 32;
      }
      return score;
    };
    std::stable_sort(moves.begin(), moves.end(), [&](const auto& a, const auto& b) {
      return priority(a) > priority(b);
    });
    return moves;
  }

  // Specialized one-ply base: defender replies need only a legal royal
  // capture check; no further checking-move tree is generated.
  Proof mateInOne() {
    Proof out;
    if (startAttack(out)) return out;
    bool unknown = false;
    for (const auto& check : checks()) {
      if (!limits_.poll()) return {};
      Played played(*this, check);
      const auto child = defend<0>();
      if (limits_.cancelled) return {};
      if (child.result == Result::kMate) return {Result::kMate, 1 + child.plies};
      unknown |= child.result == Result::kUnknown;
    }
    return unknown ? Proof{} : Proof{Result::kNoMate, 0};
  }

  // Specialized three-ply path: every defense calls the one-ply base
  // directly. Depth-five/seven instantiations terminate at this path.
  Proof mateInThree() { return checkingAttack<3>(); }

  template <int Depth>
  Proof checkingAttack() {
    Proof out;
    if (startAttack(out)) return out;
    bool unknown = false;
    for (const auto& check : checks()) {
      if (!limits_.poll()) return {};
      Played played(*this, check);
      const auto child = defend<Depth - 1>();
      if (limits_.cancelled) return {};
      if (child.result == Result::kMate) return {Result::kMate, 1 + child.plies};
      unknown |= child.result == Result::kUnknown;
    }
    return unknown ? Proof{} : Proof{Result::kNoMate, 0};
  }

  template <int Depth>
  Proof attack() {
    if constexpr (Depth < 0) {
      Proof out;
      return startAttack(out) ? out : Proof{Result::kNoMate, 0};
    } else if constexpr (Depth == 1) return mateInOne();
    else if constexpr (Depth == 3) return mateInThree();
    else return checkingAttack<Depth>();
  }

  chu::Position root_, pos_;
  chu::Color attacker_;
  const RepetitionHistory* history_;
  const ShallowMateHistory* previous_;
  ShallowMateLimits& limits_;
  int cap_;
  Material material_;
  std::vector<chu::Move> route_;
  std::array<uint64_t, 10> keys_{};
};

template <bool FullHistory>
chu::MateAnswer probe(const chu::Position& root, const RepetitionHistory* history,
    const ShallowMateHistory* previous, int depth, ShallowMateLimits& limits, int cap) {
  chu::MateAnswer answer;
  depth = oddMateDepth(depth);
  if (depth == 0) return answer;
  const auto before = limits.nodes;
  Search<FullHistory> search(root, history, previous, limits, cap);
  Proof proof;
  for (int horizon = 1; horizon <= depth; horizon += 2) {
    proof = search.solve(horizon);
    if (proof.result == Result::kMate || limits.cancelled) break;
  }
  answer.result = proof.result;
  answer.nodes = limits.nodes - before;
  answer.matePly = proof.plies;
  return answer;
}
}  // namespace

chu::MateAnswer probeShallowMate(const chu::Position& root,
    const RepetitionHistory& history, int depth, ShallowMateLimits& limits, int cap) {
  if (!history.empty()) return probe<true>(root, &history, nullptr, depth, limits, cap);
  RepetitionHistory initial;
  initial.start(root);
  return probe<true>(root, &initial, nullptr, depth, limits, cap);
}

chu::MateAnswer probeShallowMate(const chu::Position& root,
    const std::vector<uint64_t>& previous, int depth, ShallowMateLimits& limits, int cap) {
  const ShallowMateHistory view{nullptr, &previous, previous.size()};
  return probe<false>(root, nullptr, &view, depth, limits, cap);
}

chu::MateAnswer probeShallowMate(const chu::Position& root,
    const ShallowMateHistory& previous, int depth, ShallowMateLimits& limits, int cap) {
  return probe<false>(root, nullptr, &previous, depth, limits, cap);
}
}  // namespace az
