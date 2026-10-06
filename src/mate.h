// Chu shogi checkmate solver using the BNS algorithm
// (Branch Number Search, after Okabe Fumihiro's "route branch number"
// paper), implemented from the reference at JHBR3/mate/bns.{h,cc}.
// A proof/disproof-number (df-pn) arithmetic mode sharing the same engine
// is included, mirroring the reference's kPnDn control mode.
//
// What is proved here (tsume semantics, sound for the attacker):
//   the side to move (attacker) can force the capture of the enemy royal
//   with every attacker move giving check (attacking that royal) and every
//   defender move saving it. The defender has a single royal (king); a
//   defender with multiple royals (king + prince) is reported kUnknown,
//   since sacrificing one royal lets the game continue and the
//   checking/evasion tree cannot model it.
//
// Chu shogi has no check-evasion rule, but a single-royal defender who
// fails to evade loses on the spot (the royal is captured next move), so
// ignoring non-evasions is sound. A defender move capturing the attacker's
// last royal is an immediate escape (no mate), as is a defender stalemate
// being impossible while evasions exist. The attacker may be kingless —
// tsume problems conventionally omit the attacking king — in which case
// the capture-escape never applies.

#pragma once

#include <cstdint>
#include <vector>

#include "position.h"

namespace chu {

enum class MateResult { kMate, kNoMate, kUnknown };

struct MateAnswer {
  MateResult result = MateResult::kUnknown;
  std::vector<Move> pv;  // mating line in internal move form, when kMate
  uint64_t nodes = 0;
  int matePly = 0;
};

class MateSolver {
 public:
  explicit MateSolver(size_t ttMb = 32);

  // Branch-number (BNS) arithmetic when false (default), proof/disproof
  // number (df-pn style) arithmetic when true.
  void setPnDnArithmetic(bool v) { pndn_ = v; }
  void setMaxPly(int p) { maxPly_ = p; }

  MateAnswer solve(const Position& root, size_t nodeLimit);

  // solve() without the single-royal-defender entry guard, for the PV walk:
  // a proved line may legally promote the defender's elephant mid-line, and
  // the search internals handle multi-royal defenders (mate = all royals
  // captured; the sacrifice corner taints to unknown as before).
  MateAnswer solveInternal(const Position& root, size_t nodeLimit);

 private:
  bool pndn_ = false;
  int maxPly_ = 200;
  size_t ttMask_ = 0;

  struct TTEntry {
    uint64_t tag = 0;
    uint32_t abn = 1, obn = 1;
  };
  std::vector<TTEntry> tt_;

  size_t nodes_ = 0, nodeLimit_ = 0;
  int maxPlySeen_ = 0;
  std::vector<uint64_t> path_;  // hashes along the current route

  TTEntry* probe(uint64_t hash, int ply);
  TTEntry* store(uint64_t hash, int ply);

  template <bool kOrNode, bool kPnDn>
  void searchImpl(Position& pos, uint64_t hash, int ply, uint32_t abnTh,
                  uint32_t obnTh, uint32_t& outAbn, uint32_t& outObn,
                  bool& outTaint);

  void extractPv(const Position& root, MateAnswer& out);
  // Robust fallback when the TT-based extraction cannot reproduce a valid
  // line (some proof-tree children never get a TT entry at the ply the
  // extractor needs): walk the PV by re-solving each child (verdicts only).
  void extractPvByResolve(const Position& root, MateAnswer& out,
                          size_t budget);
  // Legal move-by-move and ends in capture or no-evasion.
  bool pvValidates(const Position& root, const MateAnswer& out) const;
  bool extractPvEnabled_ = true;  // off for sub-solves inside extractPvByResolve
};

}  // namespace chu
