// History-dependent repetition protection for actual play. No NN is used.
#pragma once

#include <array>
#include <vector>
#include "../position.h"

namespace az {

struct RepetitionVerdict {
  bool fourfold = false;
  int loser = -1;  // -1 = neutral repetition; otherwise the attacking color
};

class RepetitionHistory {
 public:
  void clear() { states_.clear(); }
  bool empty() const { return states_.empty(); }
  void start(const chu::Position& position);
  // The caller has already validated the move. Stores the resulting state.
  void append(const chu::Position& before, const chu::Move& move);
  void pop() { states_.pop_back(); }
  RepetitionVerdict verdict() const;
  bool hasThreefold() const;
  int occurrences(uint64_t key) const;

 private:
  struct State {
    uint64_t key = 0;
    std::array<int,2> checks{{0,0}};
    std::array<int,2> attacks{{0,0}};
    bool lishogi = true;
  };
  std::vector<State> states_;
};

// Legal root moves after which this player loses by fourfold continuous
// attack, either immediately or after a legal opponent reply. Opponent losses
// and ordinary repetition draws are retained. Uses original board coordinates.
std::vector<chu::Move> perpetualLossMoves(const chu::Position& root,
                                        const RepetitionHistory& history,
                                        int maxPlyCap = 1000);

// Reject a nominal mate PV that reaches a historical repetition draw/loss or
// the game cap before winning. The PV uses original board coordinates.
bool repetitionSafeMateLine(const chu::Position& root,
                           const RepetitionHistory& history,
                           const std::vector<chu::Move>& pv,
                           int maxPlyCap = 1000);

}  // namespace az
