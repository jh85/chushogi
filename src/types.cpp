#include "types.h"

namespace chu {

namespace {

std::array<RoleMoves, RoleCount> buildTables() {
  std::array<RoleMoves, RoleCount> t{};

  const std::vector<Dir> orth{kU, kD, kL, kR};
  const std::vector<Dir> diag{kUL, kUR, kDL, kDR};
  const std::vector<Dir> all8{kU, kD, kL, kR, kUL, kUR, kDL, kDR};

  auto set = [&](Role r, std::vector<Dir> direct, std::vector<Dir> proj) {
    t[r].direct = std::move(direct);
    t[r].proj = std::move(proj);
  };

  set(Pawn, {kU}, {});
  set(GoBetween, {kU, kD}, {});
  set(Lance, {}, {kU});
  set(Chariot, {}, {kU, kD});
  set(SideMover, {kU, kD}, {kL, kR});
  set(VerticalMover, {kL, kR}, {kU, kD});
  set(Copper, {kU, kUL, kUR, kD}, {});
  set(Silver, {kU, kUL, kUR, kDL, kDR}, {});
  set(Leopard, {kU, kUL, kUR, kD, kDL, kDR}, {});
  set(Tiger, {kD, kDL, kDR, kL, kR, kUR, kUL}, {});  // all but straight forward
  set(Gold, {kU, kD, kL, kR, kUL, kUR}, {});
  set(Elephant, {kU, kUL, kUR, kL, kR, kDL, kDR}, {});  // all but straight back
  set(Bishop, {}, diag);
  set(Rook, {}, orth);
  set(Horse, orth, diag);   // dragon horse: bishop + 1 orthogonal
  set(Dragon, diag, orth);  // dragon king: rook + 1 diagonal
  set(Kirin, {kUL, kUR, kDL, kDR, {0, -2}, {0, 2}, {2, 0}, {-2, 0}}, {});
  set(Phoenix, {kU, kD, kL, kR, {2, -2}, {2, 2}, {-2, -2}, {-2, 2}}, {});
  set(King, all8, {});
  set(Queen, {}, all8);

  // Lion: any king step, or a jump to any square at distance 2 (including
  // knight-shaped squares). Two-step moves are handled by the move generator.
  {
    std::vector<Dir> d = all8;
    for (int dx = -2; dx <= 2; ++dx)
      for (int dy = -2; dy <= 2; ++dy)
        if (std::max(dx < 0 ? -dx : dx, dy < 0 ? -dy : dy) == 2)
          d.push_back({static_cast<int8_t>(dx), static_cast<int8_t>(dy)});
    set(Lion, d, {});
  }

  // Promoted pieces
  set(Tokin, t[Gold].direct, t[Gold].proj);
  set(WhiteHorse, {}, {kU, kUL, kUR, kD});
  set(ElephantP, t[Elephant].direct, t[Elephant].proj);
  set(Whale, {}, {kU, kD, kDL, kDR});
  set(Boar, {}, {kL, kR, kUL, kUR, kDL, kDR});
  set(Ox, {}, {kU, kD, kUL, kUR, kDL, kDR});
  set(SideMoverP, t[SideMover].direct, t[SideMover].proj);
  set(VerticalMoverP, t[VerticalMover].direct, t[VerticalMover].proj);
  set(BishopP, t[Bishop].direct, t[Bishop].proj);
  set(Stag, {kUL, kUR, kL, kR, kDL, kDR}, {kU, kD});
  set(RookP, t[Rook].direct, t[Rook].proj);
  set(HorseP, t[Horse].direct, t[Horse].proj);
  set(DragonP, t[Dragon].direct, t[Dragon].proj);
  // Horned falcon: slides everywhere except straight forward; lion power only
  // along the forward file (step, double step, jump, igui/jitto).
  set(Falcon, {kU, {0, -2}}, {kUR, kR, kDR, kD, kDL, kL, kUL});
  // Soaring eagle: rook + backward diagonals; lion power on forward diagonals.
  set(Eagle, {kUR, kUL, {-2, -2}, {2, -2}}, {kU, kR, kDR, kD, kDL, kL});
  set(LionP, t[Lion].direct, t[Lion].proj);
  set(QueenP, t[Queen].direct, t[Queen].proj);
  set(Prince, t[King].direct, t[King].proj);

  return t;
}

}  // namespace

const std::array<RoleMoves, RoleCount>& roleMoves() {
  static const std::array<RoleMoves, RoleCount> tables = buildTables();
  return tables;
}

}  // namespace chu
