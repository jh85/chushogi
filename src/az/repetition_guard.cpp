#include "repetition_guard.h"

#include <algorithm>
#include <unordered_map>
#include "../attacks.h"
#include "../movegen.h"
#include "../status.h"
#include "mcts.h"

namespace az {
namespace {

// scalashogi Chushogi.isAttacking: a check, or the moved piece can capture
// an opponent piece on a normal destination next turn. Lion-trade immunity
// is cleared for this threat test. This also covers Lishogi's perpetual chase.
bool attacking(const chu::Position& after, const chu::Move& move, bool check) {
  if (check) return true;
  chu::Position probe=after;
  const auto us=!after.sideToMove;
  probe.sideToMove=us;
  probe.lastLionCapture=chu::kNoSquare;
  for (const auto& next:chu::generateLegal(probe)) {
    if (next.from!=move.to || next.mid!=chu::kNoSquare) continue;
    const auto target=probe.board[next.to];
    if (target && chu::colorOf(target)!=us) return true;
  }
  return false;
}

}  // namespace

void RepetitionHistory::start(const chu::Position& position) {
  states_.clear();
  State state;state.key=positionKey(position);
  state.lishogi=position.rules==chu::RuleSet::Lishogi;
  states_.push_back(state);
}

void RepetitionHistory::append(const chu::Position& before, const chu::Move& move) {
  if (empty()) start(before);
  auto state=states_.back();
  auto after=before;after.apply(move);
  const int us=before.sideToMove;
  const bool check=chu::inCheck(after,after.sideToMove);
  state.checks[us]=check ? state.checks[us]+1 : 0;
  const bool attack=state.lishogi ? attacking(after,move,check) : check;
  state.attacks[us]=attack ? state.attacks[us]+1 : 0;
  state.key=positionKey(after);
  states_.push_back(state);
}

RepetitionVerdict RepetitionHistory::verdict() const {
  RepetitionVerdict result;
  if (states_.empty()) return result;
  const auto& current=states_.back();
  size_t first=states_.size()-1;
  int count=0;
  for (size_t i=0;i<states_.size();++i) {
    if (states_[i].key!=current.key) continue;
    if (count++==0) first=i;
  }
  if (count<4) return result;
  result.fourfold=true;
  const int distance=static_cast<int>((states_.size()-1-first)/2);
  const auto& runs=current.lishogi ? current.attacks : current.checks;
  const bool black=runs[chu::Sente]>=distance;
  const bool white=runs[chu::Gote]>=distance;
  if (black!=white) result.loser=black ? chu::Sente : chu::Gote;
  return result;
}

bool RepetitionHistory::hasThreefold() const {
  std::unordered_map<uint64_t,int> counts;
  for (const auto& state:states_) if (++counts[state.key]>=3) return true;
  return false;
}

std::vector<chu::Move> perpetualLossMoves(const chu::Position& root,
                                        const RepetitionHistory& history,
                                        int maxPlyCap) {
  std::vector<chu::Move> excluded;
  if (!history.hasThreefold()) return excluded;
  auto path=history;
  const auto us=root.sideToMove;
  for (const auto& move:chu::generateLegal(root)) {
    auto child=root;child.apply(move);
    if (chu::evaluateStatus(child).end!=chu::GameEnd::Playing) continue;
    path.append(root,move);
    const auto immediate=path.verdict();
    bool losing=immediate.fourfold && immediate.loser==us;
    // A neutral draw or the opponent's foul ends the game; do not search
    // a fictitious reply after it. Likewise the actual game ply cap.
    if (!immediate.fourfold && child.moveNumber-1<maxPlyCap) {
      for (const auto& reply:chu::generateLegal(child)) {
        auto after=child;after.apply(reply);
        if (chu::evaluateStatus(after).end!=chu::GameEnd::Playing) continue;
        path.append(child,reply);
        const auto outcome=path.verdict();path.pop();
        if (outcome.fourfold && outcome.loser==us) {losing=true;break;}
      }
    }
    path.pop();
    if (losing) excluded.push_back(move);
  }
  return excluded;
}

bool repetitionSafeMateLine(const chu::Position& root,
                           const RepetitionHistory& history,
                           const std::vector<chu::Move>& pv, int maxPlyCap) {
  auto path=history;auto position=root;
  for (const auto& move:pv) {
    path.append(position,move);position.apply(move);
    const auto status=chu::evaluateStatus(position);
    if (status.end!=chu::GameEnd::Playing) return status.winner==root.sideToMove;
    const auto outcome=path.verdict();
    if (outcome.fourfold) return outcome.loser==!root.sideToMove;
    if (position.moveNumber-1>=maxPlyCap) return false;
  }
  return !pv.empty();
}

}  // namespace az
