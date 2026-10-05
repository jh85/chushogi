#include "movegen.h"

#include "attacks.h"

namespace chu {

std::vector<Move> generateLegal(const Position& pos) {
  std::vector<Move> out;
  const Color us = pos.sideToMove;
  const Color them = !us;
  const int llc = pos.lastLionCapture;

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
      if (llc != kNoSquare) {
        uint8_t occ = pos.board[d];
        if (d != llc && occ && colorOf(occ) == them && isLion(roleOf(occ)))
          return false;
      }
      return true;
    };
    for (int d : shortDests)
      if (keepSingle(d)) dests.push_back(d);
    for (int d : slideDests)
      if (keepSingle(d)) dests.push_back(d);

    // Ordinary and promoting single moves.
    for (int d : dests) {
      out.push_back({static_cast<int16_t>(s), kNoSquare,
                     static_cast<int16_t>(d), false});
      if (canPromote(role, us, s, d, pos.board[d] != 0))
        out.push_back({static_cast<int16_t>(s), kNoSquare,
                       static_cast<int16_t>(d), true});
    }

    // Two-step moves of lion-power pieces.
    if (!hasLionPower(role)) continue;
    for (int ms : shortDests) {
      if (dist(s, ms) != 1) continue;  // first step must be adjacent
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
          if (llc != kNoSquare && d != llc && occ && colorOf(occ) == them &&
              isLion(roleOf(occ)))
            continue;
          out.push_back({static_cast<int16_t>(s), static_cast<int16_t>(ms),
                         static_cast<int16_t>(d), false});
        }
      }
    }
  }
  return out;
}

bool isLegal(const Position& pos, const Move& m) {
  for (const Move& g : generateLegal(pos))
    if (g.from == m.from && g.mid == m.mid && g.to == m.to &&
        g.promote == m.promote)
      return true;
  return false;
}

}  // namespace chu
