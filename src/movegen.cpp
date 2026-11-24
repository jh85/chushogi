#include "movegen.h"

#include "attacks.h"

namespace chu {

namespace {

// ---- Rule differences between lishogi and JCSA (docs/jcsa/jcsa-delta.md,
// rows 1-3). The generator calls them with its RuleSet parameter.

// R-P3, R-P4: who may promote on reaching the last rank without entering the
// zone or capturing. lishogi: the pawn and the lance. JCSA: the pawn only; a
// lance that reaches the last rank unpromoted becomes a dead piece (the lance
// relief of R-P5 is for correspondence games only and is not applied).
template <RuleSet R>
bool lastRankPromotion(Role role) {
  if constexpr (R == RuleSet::JCSA)
    return role == Pawn;
  else
    return role == Pawn || role == Lance;
}

// canPromote() under rule set R.
template <RuleSet R>
bool mayPromote(Role role, Color c, int from, int to, bool capture) {
  if (promoteRole(role) < 0) return false;
  if (zonePromotion(c, from, to, capture)) return true;
  return lastRankPromotion<R>(role) && sqY(to) == backrankY(c);
}

// U2, U3: the lion on `sq` has a foot if a piece of its own side protects
// that square. Judged at the attempted capture, with the capturing piece
// lifted from `from` (X-rays through it count), for each lion separately.
bool lionHasFoot(const Position& pos, int sq, int from) {
  return threatened(pos, colorOf(pos.board[sq]), sq, /*pawnsOnly=*/false,
                    from, kNoSquare);
}

// R-L4, H-3⑥: the counter-strike ban (sakishishi). After a non-lion captured
// a lion on pos.lastLionCapture, a non-lion moving from `from` may not take
// an enemy lion on `sq`. lishogi exempts the capture square itself (a kirin
// that took the lion there and promoted) and bans every other such capture;
// JCSA (delta row 1b, U4) has no capture-square exemption and bans the
// capture of every lion with a foot. Lions never consult this: JCSA's ban
// binds them too (U1), but its exceptions (an adjacent lion, tsukegui) leave
// only captures that the lion-trading rules (R-L2, R-L3) already forbid.
template <RuleSet R>
bool counterStrikeBanned(const Position& pos, int from, int sq) {
  const int llc = pos.lastLionCapture;
  if (llc == kNoSquare) return false;
  if (R == RuleSet::Lishogi && sq == llc) return false;
  const uint8_t occ = pos.board[sq];
  if (!occ || colorOf(occ) == pos.sideToMove || !isLion(roleOf(occ)))
    return false;
  if constexpr (R == RuleSet::JCSA)
    return lionHasFoot(pos, sq, from);
  else
    return true;
}

template <RuleSet R, class Sink>
void generate(const Position& pos, Sink& out) {
  const Color us = pos.sideToMove;
  const Color them = !us;

  for (int s = 0; s < kSquares; ++s) {
    const uint8_t pc = pos.board[s];
    if (!pc || colorOf(pc) != us) continue;
    const Role role = roleOf(pc);
    const RoleMoves& rm = roleMoves()[role];
    const bool lion = isLion(role);

    // Direct moves (steps and jumps).
    std::vector<int> shortDests;
    for (Dir d : rm.direct) {
      Dir p = physical(d, us);
      int x = sqX(s) + p.dx, y = sqY(s) + p.dy;
      if (!onBoard(x, y)) continue;
      int t = makeSq(x, y);
      uint8_t occ = pos.board[t];
      if (!occ || colorOf(occ) != us) shortDests.push_back(t);
    }
    // Sliding moves.
    std::vector<int> slideDests;
    for (Dir d : rm.proj) {
      Dir u = physical(d, us);
      int x = sqX(s) + u.dx, y = sqY(s) + u.dy;
      while (onBoard(x, y)) {
        int t = makeSq(x, y);
        uint8_t occ = pos.board[t];
        if (!occ) {
          slideDests.push_back(t);
        } else {
          if (colorOf(occ) != us) slideDests.push_back(t);
          break;
        }
        x += u.dx;
        y += u.dy;
      }
    }

    // moveFilter: lion-capture restrictions on single (one-destination) moves.
    std::vector<int> dests;
    dests.reserve(shortDests.size() + slideDests.size());
    auto keepSingle = [&](int d) {
      if (lion) {
        uint8_t occ = pos.board[d];
        if (occ && colorOf(occ) == them && isLion(roleOf(occ)) &&
            dist(s, d) > 1 &&
            threatened(pos, them, d, /*pawnsOnly=*/false, s, kNoSquare))
          return false;
        return true;
      }
      return !counterStrikeBanned<R>(pos, s, d);
    };
    for (int d : shortDests)
      if (keepSingle(d)) dests.push_back(d);
    for (int d : slideDests)
      if (keepSingle(d)) dests.push_back(d);

    // Ordinary and promoting single moves.
    for (int d : dests) {
      out.push_back({static_cast<int16_t>(s), kNoSquare,
                     static_cast<int16_t>(d), false});
      if (mayPromote<R>(role, us, s, d, pos.board[d] != 0))
        out.push_back({static_cast<int16_t>(s), kNoSquare,
                       static_cast<int16_t>(d), true});
    }

    // Two-step moves of lion-power pieces.
    if (!hasLionPower(role)) continue;
    for (int ms : shortDests) {
      if (dist(s, ms) != 1) continue;  // first step must be adjacent
      // R-L4 (delta row 2): the ban covers every step of a double move. A
      // horned falcon or soaring eagle that may not take the lion on the mid
      // square has no move through it, as in lishogi since scalashogi
      // fcf0664 ("Prevent illegal lion recapture in midstep", 2026-10-23;
      // before it, lishogi tested only the destination). A lion may always
      // take an adjacent lion.
      if (!lion && counterStrikeBanned<R>(pos, s, ms)) continue;
      // Returning to the origin (igui after a capture, or jitto pass) is
      // always included once the mid step itself is playable.
      out.push_back({static_cast<int16_t>(s), static_cast<int16_t>(ms),
                     static_cast<int16_t>(s), false});
      for (int d : shortDests) {
        if (dist(d, ms) != 1) continue;
        const uint8_t occ = pos.board[d];
        if (lion) {
          if (!occ) {
            out.push_back({static_cast<int16_t>(s), static_cast<int16_t>(ms),
                           static_cast<int16_t>(d), false});
            continue;
          }
          // Capturing on the second step.
          if (dist(d, s) == 1 || !isLion(roleOf(occ))) {
            out.push_back({static_cast<int16_t>(s), static_cast<int16_t>(ms),
                           static_cast<int16_t>(d), false});
            continue;
          }
          // Capturing a non-adjacent enemy lion.
          const uint8_t midOcc = pos.board[ms];
          const bool midSubstantial =
              midOcc && roleOf(midOcc) != Pawn && roleOf(midOcc) != GoBetween;
          if (midSubstantial ||
              (!threatened(pos, them, d, /*pawnsOnly=*/false, s, ms) &&
               !threatened(pos, them, d, /*pawnsOnly=*/true, s, kNoSquare)))
            out.push_back({static_cast<int16_t>(s), static_cast<int16_t>(ms),
                           static_cast<int16_t>(d), false});
        } else {
          // Horned falcon / soaring eagle: second-step destinations are only
          // restricted by the lion-recapture ban.
          if (counterStrikeBanned<R>(pos, s, d)) continue;
          out.push_back({static_cast<int16_t>(s), static_cast<int16_t>(ms),
                         static_cast<int16_t>(d), false});
        }
      }
    }
  }
}

struct RoyalCaptureSink {
  int royal;
  std::vector<Move> moves;
  void push_back(const Move& move) {
    if (move.to == royal || move.mid == royal) moves.push_back(move);
  }
};

struct CheckingSink {
  Position work;
  int royal;
  std::vector<int> candidates;
  std::vector<Move> moves;

  CheckingSink(const Position& pos, int target) : work(pos), royal(target) {
    // Other pieces can give discovered checks only along sliding lines.
    // Keep existing direct checkers too (chu permits a previous non-evasion).
    for (int sq = 0; sq < kSquares; ++sq) {
      const auto piece = pos.board[sq];
      if (!piece || colorOf(piece) != pos.sideToMove) continue;
      bool potential = pieceAttacks(pos, sq, royal, pos.sideToMove,
                                    kNoSquare, kNoSquare);
      const int dx = sqX(royal) - sqX(sq), dy = sqY(royal) - sqY(sq);
      for (auto dir : roleMoves()[roleOf(piece)].proj) {
        const auto p = physical(dir, pos.sideToMove);
        potential |= dx * p.dy == dy * p.dx && dx * p.dx + dy * p.dy > 0;
      }
      if (potential) candidates.push_back(sq);
    }
  }

  void push_back(const Move& move) {
    if (move.to == royal || move.mid == royal) {
      moves.push_back(move);
      return;
    }
    const auto us = work.sideToMove;
    const MoveUndo undo(work, move);
    work.apply(move);
    bool check = pieceAttacks(work, move.to, royal, us, kNoSquare, kNoSquare);
    if (!check) for (int sq : candidates) {
      if (sq == move.from) continue;
      if (pieceAttacks(work, sq, royal, us, kNoSquare, kNoSquare)) {
        check = true;
        break;
      }
    }
    undo.restore(work);
    if (check) moves.push_back(move);
  }
};

template <class Sink>
void dispatch(const Position& pos, Sink& sink) {
  if (pos.rules == RuleSet::JCSA) generate<RuleSet::JCSA>(pos, sink);
  else generate<RuleSet::Lishogi>(pos, sink);
}

}  // namespace

std::vector<Move> generateLegal(const Position& pos) {
  std::vector<Move> moves;
  dispatch(pos, moves);
  return moves;
}

std::vector<Move> generateChecking(const Position& pos, int royalSquare) {
  CheckingSink sink(pos, royalSquare);
  dispatch(pos, sink);
  return std::move(sink.moves);
}

std::vector<Move> generateRoyalCaptures(const Position& pos, int royalSquare) {
  RoyalCaptureSink sink{royalSquare, {}};
  dispatch(pos, sink);
  return std::move(sink.moves);
}

bool isLegal(const Position& pos, const Move& m) {
  for (const Move& g : generateLegal(pos))
    if (g.from == m.from && g.mid == m.mid && g.to == m.to &&
        g.promote == m.promote)
      return true;
  return false;
}

}  // namespace chu
