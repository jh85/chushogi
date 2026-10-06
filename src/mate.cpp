#include "mate.h"

#include "sfen.h"
#include "usi.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "attacks.h"
#include "movegen.h"

namespace chu {

// =====================================================================
// Pure branch-number arithmetic, ported from JHBR3/mate/bns.h
// =====================================================================

namespace bns {

constexpr uint32_t kInf = 1u << 30;

inline uint32_t SatAdd(uint32_t a, uint32_t b) {
  uint64_t s = static_cast<uint64_t>(a) + b;
  return s >= kInf ? kInf : static_cast<uint32_t>(s);
}

struct ChildView {
  uint32_t abn = 1, obn = 1;
};

struct Summary {
  uint32_t abn = 1, obn = 1;
  int best = -1;
  uint32_t second = kInf;
  uint32_t k = 0;
  bool proved = false, disproved = false;
  bool terminal() const { return proved || disproved; }
};

template <bool kOrNode, bool kPnDn>
inline Summary Summarize(const ChildView* c, int n) {
  Summary s;
  uint32_t best_rel = kInf, best_opp = kInf, second = kInf, k = 0;
  uint64_t opp_sum = 0;
  int best = -1;

  for (int i = 0; i < n; i++) {
    const uint32_t rel = kOrNode ? c[i].abn : c[i].obn;
    const uint32_t opp = kOrNode ? c[i].obn : c[i].abn;
    if (rel == 0) {
      s.best = i;
      s.proved = kOrNode;
      s.disproved = !kOrNode;
      if (kOrNode) { s.abn = 0; s.obn = kInf; }
      else { s.abn = kInf; s.obn = 0; }
      return s;
    }
    if (kPnDn) opp_sum += opp;
    if (rel >= kInf) continue;
    k++;
    if (rel < best_rel || (kPnDn && rel == best_rel && opp < best_opp)) {
      second = best_rel;
      best_rel = rel;
      best_opp = opp;
      best = i;
    } else if (rel < second) {
      second = rel;
    }
  }

  const bool exhausted = kPnDn ? opp_sum == 0 : k == 0;
  if (exhausted) {
    s.proved = !kOrNode;
    s.disproved = kOrNode;
    if (kOrNode) { s.abn = kInf; s.obn = 0; }
    else { s.abn = 0; s.obn = kInf; }
    return s;
  }

  const uint32_t node_rel = best_rel;
  const uint32_t node_opp =
      kPnDn ? (opp_sum >= kInf ? kInf : static_cast<uint32_t>(opp_sum))
            : SatAdd(best_opp, k - 1);
  s.abn = kOrNode ? node_rel : node_opp;
  s.obn = kOrNode ? node_opp : node_rel;
  s.best = best;
  s.second = second;
  s.k = k;
  return s;
}

template <bool kOrNode>
inline void ChildThresholds(const Summary& s, const ChildView& best,
                            uint32_t abn_th, uint32_t obn_th,
                            uint32_t* child_abn_th, uint32_t* child_obn_th) {
  const uint32_t rel_th = kOrNode ? abn_th : obn_th;
  const uint32_t opp_th = kOrNode ? obn_th : abn_th;
  const uint32_t node_opp = kOrNode ? s.obn : s.abn;
  const uint32_t best_opp = kOrNode ? best.obn : best.abn;
  const uint32_t rel_out =
      rel_th < SatAdd(s.second, 1) ? rel_th : SatAdd(s.second, 1);
  const uint32_t opp_out =
      opp_th >= kInf ? kInf : opp_th - (node_opp - best_opp);
  *child_abn_th = kOrNode ? rel_out : opp_out;
  *child_obn_th = kOrNode ? opp_out : rel_out;
}

constexpr ChildView MateView() { return ChildView{0, kInf}; }
constexpr ChildView NoMateView() { return ChildView{kInf, 0}; }

}  // namespace bns

// =====================================================================
// Zobrist hashing
// =====================================================================

namespace {

uint64_t g_zobrist[80][kSquares];
uint64_t g_zside[2];
uint64_t g_zllc[kSquares + 1];
bool g_zinit = false;

void initZobrist() {
  if (g_zinit) return;
  uint64_t seed = 0xC0FFEE123456789ull;
  auto next = [&]() {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return seed;
  };
  for (int p = 0; p < 80; ++p)
    for (int s = 0; s < kSquares; ++s) g_zobrist[p][s] = next();
  g_zside[0] = next();
  g_zside[1] = next();
  for (int i = 0; i <= kSquares; ++i) g_zllc[i] = next();
  g_zinit = true;
}

uint64_t hashPosition(const Position& pos) {
  uint64_t h = g_zside[pos.sideToMove];
  for (int s = 0; s < kSquares; ++s) {
    uint8_t p = pos.board[s];
    if (p) h ^= g_zobrist[p][s];
  }
  h ^= g_zllc[pos.lastLionCapture == kNoSquare ? kSquares : pos.lastLionCapture];
  return h;
}

int royalCount(const Position& pos, Color c) {
  int n = 0;
  for (int s = 0; s < kSquares; ++s) {
    uint8_t p = pos.board[s];
    if (p && colorOf(p) == c && isRoyal(roleOf(p))) ++n;
  }
  return n;
}

// Is any royal of `c` attacked by the opponent?
bool royalAttacked(const Position& pos, Color c) {
  for (int s = 0; s < kSquares; ++s) {
    uint8_t p = pos.board[s];
    if (p && colorOf(p) == c && isRoyal(roleOf(p)) &&
        threatened(pos, !c, s, /*pawnsOnly=*/false, kNoSquare, kNoSquare))
      return true;
  }
  return false;
}

// One node in the search: a move plus its cached child state.
struct Child {
  Move move;
  uint64_t hash = 0;
  bns::ChildView view;
  bool pinned = false;   // view fixed locally; never re-probe the TT
  bool rTaint = false;   // resource-tainted no-mate (depth cap, multi-royal)
};

constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ull;

}  // namespace

// =====================================================================
// MateSolver
// =====================================================================

MateSolver::MateSolver(size_t ttMb) {
  initZobrist();
  size_t want = (ttMb << 20) / sizeof(TTEntry);
  size_t n = size_t{1} << 16;
  while (n * 2 <= want) n *= 2;
  ttMask_ = n - 1;
  tt_.resize(n);
}

MateSolver::TTEntry* MateSolver::probe(uint64_t hash, int ply) {
  const uint64_t mixed = hash + static_cast<uint64_t>(ply) * kGolden;
  TTEntry& e = tt_[mixed & ttMask_];
  return e.tag == mixed ? &e : nullptr;
}

MateSolver::TTEntry* MateSolver::store(uint64_t hash, int ply) {
  const uint64_t mixed = hash + static_cast<uint64_t>(ply) * kGolden;
  TTEntry& e = tt_[mixed & ttMask_];
  e.tag = mixed;
  return &e;
}

template <bool kOrNode, bool kPnDn>
void MateSolver::searchImpl(Position& pos, uint64_t hash, int ply,
                            uint32_t abnTh, uint32_t obnTh, uint32_t& outAbn,
                            uint32_t& outObn, bool& outTaint) {
  ++nodes_;
  if (ply > maxPlySeen_) maxPlySeen_ = ply;
  const Color us = pos.sideToMove;
  const Color them = !us;

  if (TTEntry* e = probe(hash, ply)) {
    if (e->abn == 0 || e->obn == 0) {
      outAbn = e->abn;
      outObn = e->obn;
      outTaint = false;
      return;
    }
  }

  // ---- Move filtering / node terminals (see mate.h for the model) ----
  std::vector<Child> children;
  const std::vector<Move> legal = generateLegal(pos);
  if constexpr (kOrNode) {
    // Attacker: checking moves only; a move capturing the last enemy royal
    // is an immediate win.
    for (const Move& m : legal) {
      Position c = pos;
      c.apply(m);
      if (royalCount(c, them) == 0) {
        TTEntry* e = store(hash, ply);
        e->abn = 0;
        e->obn = bns::kInf;
        outAbn = 0;
        outObn = bns::kInf;
        outTaint = false;
        return;
      }
      if (royalAttacked(c, them))
        children.push_back(Child{m, hashPosition(c), {}, false, false});
    }
    if (children.empty()) {  // no checks: no restricted mate
      TTEntry* e = store(hash, ply);
      e->abn = bns::kInf;
      e->obn = 0;
      outAbn = bns::kInf;
      outObn = 0;
      outTaint = false;
      return;
    }
  } else {
    if (legal.empty()) {  // stalemate: the defender loses
      TTEntry* e = store(hash, ply);
      e->abn = 0;
      e->obn = bns::kInf;
      outAbn = 0;
      outObn = bns::kInf;
      outTaint = false;
      return;
    }
    // Defender: evasions (no own royal attacked after the move), plus any
    // move capturing the attacker's last royal (an immediate escape). A
    // kingless attacker (tsume convention) can never be captured out.
    for (const Move& m : legal) {
      Position c = pos;
      c.apply(m);
      const uint64_t chash = hashPosition(c);
      if (royalCount(pos, them) > 0 && royalCount(c, them) == 0) {
        Child child{m, chash, bns::NoMateView(), true, false};
        children.push_back(child);
        continue;
      }
      if (!royalAttacked(c, us))
        children.push_back(Child{m, chash, {}, false, false});
    }
    if (children.empty()) {
      if (royalCount(pos, us) == 1) {
        // No way to save the only royal: checkmate.
        TTEntry* e = store(hash, ply);
        e->abn = 0;
        e->obn = bns::kInf;
        outAbn = 0;
        outObn = bns::kInf;
        outTaint = false;
        return;
      }
      // Multi-royal defender can sacrifice a royal and play on: the
      // restricted tree cannot model it; report a tainted no-mate (which
      // becomes kUnknown at the root).
      outAbn = bns::kInf;
      outObn = 0;
      outTaint = true;
      return;
    }
  }

  // ---- Threshold iteration (JHBR3 SearchImpl main loop) ----
  std::vector<bns::ChildView> views(children.size());
  const int n = static_cast<int>(children.size());
  int refreshOnly = -1;
  for (;;) {
    const int lo = refreshOnly < 0 ? 0 : refreshOnly;
    const int hi = refreshOnly < 0 ? n : refreshOnly + 1;
    for (int i = lo; i < hi; ++i) {
      if (children[i].pinned) {
        views[i] = children[i].view;
        continue;
      }
      if (TTEntry* e = probe(children[i].hash, ply + 1)) {
        views[i] = {e->abn, e->obn};
      } else {
        views[i] = {1, 1};
      }
    }
    const bns::Summary s = bns::Summarize<kOrNode, kPnDn>(views.data(), n);

    if (s.terminal()) {
      bool taint = false;
      if (s.disproved) {
        if (s.best >= 0)
          taint = children[s.best].rTaint;
        else
          for (const Child& ch : children) taint = taint || ch.rTaint;
      }
      if (!taint) {
        TTEntry* e = store(hash, ply);
        e->abn = s.abn;
        e->obn = s.obn;
      }
      outAbn = s.abn;
      outObn = s.obn;
      outTaint = taint;
      return;
    }

    {
      TTEntry* e = store(hash, ply);
      e->abn = s.abn;
      e->obn = s.obn;
    }

    if (abnTh <= s.abn || obnTh <= s.obn || nodes_ >= nodeLimit_) {
      outAbn = s.abn;
      outObn = s.obn;
      outTaint = false;
      return;
    }

    uint32_t cthA, cthB;
    bns::ChildThresholds<kOrNode>(s, views[s.best], abnTh, obnTh, &cthA,
                                  &cthB);
    Child& ch = children[s.best];

    // Route-dependent child terminals: depth cap, then path cycle.
    if (ply + 1 >= maxPly_) {
      ch.view = bns::NoMateView();
      ch.pinned = true;
      ch.rTaint = true;
      refreshOnly = s.best;
      continue;
    }
    bool onPath = false;
    for (uint64_t h : path_)
      if (h == ch.hash) {
        onPath = true;
        break;
      }
    if (onPath) {
      ch.view = bns::NoMateView();
      ch.pinned = true;
      ch.rTaint = false;
      refreshOnly = s.best;
      continue;
    }

    Position cpos = pos;
    cpos.apply(ch.move);
    path_.push_back(ch.hash);
    uint32_t ra, rb;
    bool rt;
    searchImpl<!kOrNode, kPnDn>(cpos, ch.hash, ply + 1, cthA, cthB, ra, rb,
                                rt);
    path_.pop_back();
    ch.view = {ra, rb};
    ch.rTaint = rt;
    ch.pinned = true;  // keep returned values; tainted ones are not in the TT
    refreshOnly = s.best;
  }
}

bool MateSolver::pvValidates(const Position& root, const MateAnswer& out)
    const {
  const Color attacker = root.sideToMove;
  const Color defender = !attacker;
  Position verify = root;
  for (const Move& m : out.pv) {
    if (!isLegal(verify, m)) return false;
    verify.apply(m);
  }
  if (royalCount(verify, defender) > 0) {
    // Otherwise the pv must end with the defender to move and no evasion.
    if (verify.sideToMove != defender) return false;
    for (const Move& m : generateLegal(verify)) {
      Position c = verify;
      c.apply(m);
      const bool capturedOut =
          royalCount(verify, attacker) > 0 && royalCount(c, attacker) == 0;
      if (capturedOut || !royalAttacked(c, defender))
        return false;  // the defender escapes: the pv is not a mate
    }
  }
  return true;
}

void MateSolver::extractPvByResolve(const Position& root, MateAnswer& out,
                                    size_t budget) {
  // Greedy re-solve walk, replicating the search's AND/OR alternation.
  // Attacker nodes: the fastest-proving checking child (fewest nodes).
  // Defender nodes: any evasion whose child is still a proven mate.
  // The mate ends with the royal captured or the defender out of evasions.
  // Transpositions are allowed (tsume lines can drive a king around); the
  // walk stalls only if a position occurs for the third time.
  const Color attacker = root.sideToMove;
  const Color defender = !attacker;
  const size_t perChild = budget;
  extractPvEnabled_ = false;

  // Does this defender-to-move position still lose on every line?
  // (AND node: all evasions must fail.)
  auto allEvasionsLose = [&](const Position& dpos) {
    for (const Move& e : generateLegal(dpos)) {
      Position c2 = dpos;
      c2.apply(e);
      if (royalCount(dpos, attacker) > 0 && royalCount(c2, attacker) == 0)
        return false;  // attacker captured out: refutation
      if (royalAttacked(c2, defender))
        continue;  // not an evasion
      if (solveInternal(c2, perChild).result != MateResult::kMate)
        return false;
    }
    return true;  // no evasion saves the royal: mated
  };

  Position pos = root;
  std::vector<Move> pv;
  std::vector<uint64_t> seen{hashPosition(pos)};
  auto repeats = [&](uint64_t h) {
    return std::count(seen.begin(), seen.end(), h) >= 2;
  };
  for (int ply = 0; ply < 2 * maxPly_; ++ply) {
    const bool attackerTurn = pos.sideToMove == attacker;
    Move best;
    size_t bestNodes = 0;
    bool found = false, done = false;
    if (!attackerTurn) {
      bool anyEvasion = false;
      for (const Move& m : generateLegal(pos)) {
        Position c = pos;
        c.apply(m);
        if (royalCount(pos, attacker) > 0 && royalCount(c, attacker) == 0)
          continue;  // would refute the proof: never on a proved node
        if (royalAttacked(c, defender))
          continue;  // not an evasion
        anyEvasion = true;
        const uint64_t h = hashPosition(c);
        if (repeats(h)) continue;
        if (solveInternal(c, perChild).result != MateResult::kMate) continue;
        best = m;
        found = true;
        break;
      }
      if (!anyEvasion) break;  // mate complete: defender has no evasion
    } else {
      for (const Move& m : generateLegal(pos)) {
        Position c = pos;
        c.apply(m);
        if (royalCount(c, defender) == 0) {  // capture the last royal
          best = m;
          found = done = true;
          break;
        }
        if (!royalAttacked(c, defender)) continue;  // must give check
        const uint64_t h = hashPosition(c);
        if (repeats(h)) continue;
        if (!allEvasionsLose(c)) continue;
        // prefer the fastest-proving line (fewest nodes): the remaining mate
        // distance shrinks, so the walk cannot circle indefinitely
        size_t n = 0;
        {  // one more sub-solve for the node count of this accepted child
          Position c2 = pos;
          c2.apply(m);
          // allEvasionsLose already re-solved the grandchildren; for the
          // child itself, its own verdict is mate by construction here
          MateAnswer sub = solveInternal(c2, perChild);
          n = sub.nodes;
        }
        if (!found || n < bestNodes) {
          best = m;
          bestNodes = n;
          found = true;
        }
      }
    }
    if (!found) {
      pv.clear();
      break;
    }
    pv.push_back(best);
    pos.apply(best);
    const uint64_t h = hashPosition(pos);
    if (repeats(h)) {  // third occurrence: not a clean PV
      pv.clear();
      break;
    }
    seen.push_back(h);
    if (done) break;
  }
  extractPvEnabled_ = true;
  out.pv = std::move(pv);
  out.matePly = static_cast<int>(out.pv.size());
}

void MateSolver::extractPv(const Position& root, MateAnswer& out) {
  Position pos = root;
  std::vector<Move> pv;
  const Color attacker = root.sideToMove;
  const bool attackerKingless = royalCount(root, attacker) == 0;
  for (int ply = 0; ply < 2 * maxPly_; ++ply) {
    const Color us = pos.sideToMove;
    const Color them = !us;
    // The kingless attacker of a tsume problem can never be captured out.
    if (royalCount(pos, them) == 0 &&
        !(them == attacker && attackerKingless))
      break;
    Move chosen;
    bool found = false;
    for (const Move& m : generateLegal(pos)) {
      Position c = pos;
      c.apply(m);
      if (royalCount(pos, them) > 0 &&
          royalCount(c, them) == 0) {  // immediate capture
        chosen = m;
        found = true;
        break;
      }
      if (TTEntry* e = probe(hashPosition(c), ply + 1)) {
        if (e->abn == 0) {  // proved child
          chosen = m;
          found = true;
          break;
        }
      }
    }
    if (!found) break;
    pv.push_back(chosen);
    pos.apply(chosen);
  }
  out.pv = std::move(pv);
  out.matePly = static_cast<int>(out.pv.size());
}

MateAnswer MateSolver::solve(const Position& root, size_t nodeLimit) {
  MateAnswer out;
  if (royalCount(root, !root.sideToMove) != 1) {
    // Only single-royal defenders are supported (see mate.h).
    return out;
  }
  return solveInternal(root, nodeLimit);
}

MateAnswer MateSolver::solveInternal(const Position& root, size_t nodeLimit) {
  std::fill(tt_.begin(), tt_.end(), TTEntry{});
  path_.clear();
  nodes_ = 0;
  nodeLimit_ = nodeLimit;
  maxPlySeen_ = 0;

  MateAnswer out;
  const Color attacker = root.sideToMove;
  const Color defender = !attacker;
  // The attacker may be kingless: tsume problems conventionally omit the
  // attacking king, and such an attacker can never be captured out.

  Position pos = root;
  const uint64_t h = hashPosition(pos);
  path_.push_back(h);
  uint32_t ra, rb;
  bool rt;
  if (pndn_)
    searchImpl<true, true>(pos, h, 0, bns::kInf, bns::kInf, ra, rb, rt);
  else
    searchImpl<true, false>(pos, h, 0, bns::kInf, bns::kInf, ra, rb, rt);

  out.nodes = nodes_;
  if (ra == 0) {
    out.result = MateResult::kMate;
    if (extractPvEnabled_) {
      extractPv(root, out);
      if (!pvValidates(root, out)) {
        // The TT-based walk can stall (the search resolves some children
        // via pinned or transposed paths that never land an entry at the
        // ply the extractor needs). Re-extract by re-solving: verdict-only
        // sub-solves, correct by construction.
        extractPvByResolve(root, out, nodeLimit);
      }
      if (!pvValidates(root, out)) {
        std::fprintf(stderr, "error: mate pv failed self-validation: %s\n",
                     toSfen(root).c_str());
        out.result = MateResult::kUnknown;
        out.pv.clear();
        out.matePly = 0;
      }
    }
  } else if (rb == 0 && !rt) {
    out.result = MateResult::kNoMate;
  }
  return out;
}

}  // namespace chu
