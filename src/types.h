// Chu Shogi (12x12) core types and piece movement tables.
//
// Board coordinates: square index = y * 12 + x, where
//   x = file index  (file 1 -> 0 ... file 12 -> 11)
//   y = rank index  (rank a -> 0 ... rank l -> 11)
// This matches lishogi/scalashogi: square key = <file number><rank letter>,
// e.g. 1a is the top-right corner from Sente's point of view, 12l the
// bottom-left.
//
// Directions below are written from Sente's point of view:
//   up = decreasing rank index (towards rank a), left = increasing file index.
// For Gote every dy is mirrored.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace chu {

enum Color : uint8_t { Sente = 0, Gote = 1 };
inline Color operator!(Color c) { return c == Sente ? Gote : Sente; }

// Roles, following scalashogi's naming. "*P" roles are promoted versions that
// are distinct from the identical-moving original piece (e.g. RookP is a rook
// obtained from a promoted gold general: it cannot promote further).
enum Role : uint8_t {
  // Unpromoted pieces of the initial setup
  Pawn = 0, Lance, GoBetween, Chariot, SideMover, VerticalMover,
  Copper, Silver, Leopard, Tiger, Gold, Elephant,
  Bishop, Rook, Horse, Dragon, Kirin, Phoenix,
  King, Lion, Queen,
  // Pieces appearing only through promotion
  Tokin,        // +p (promoted pawn, moves as gold)
  WhiteHorse,   // +l
  ElephantP,    // +i (drunk elephant from go-between; cannot promote again)
  Whale,        // +a
  Boar,         // +m
  Ox,           // +v
  SideMoverP,   // +c (side mover from copper)
  VerticalMoverP, // +s (vertical mover from silver)
  BishopP,      // +f (bishop from leopard)
  Stag,         // +t
  RookP,        // +g (rook from gold)
  HorseP,       // +b (dragon horse from bishop)
  DragonP,      // +r (dragon king from rook)
  Falcon,       // +h (horned falcon from dragon horse)
  Eagle,        // +d (soaring eagle from dragon king)
  LionP,        // +o (lion from kirin)
  QueenP,       // +x (free king from phoenix)
  Prince,       // +e
  RoleCount
};

constexpr int kFiles = 12;
constexpr int kRanks = 12;
constexpr int kSquares = kFiles * kRanks;
constexpr int kNoSquare = -1;

inline int sqX(int sq) { return sq % kFiles; }
inline int sqY(int sq) { return sq / kFiles; }
inline int makeSq(int x, int y) { return y * kFiles + x; }
inline bool onBoard(int x, int y) {
  return 0 <= x && x < kFiles && 0 <= y && y < kRanks;
}
inline bool onBoard(int sq) { return 0 <= sq && sq < kSquares; }
// Chebyshev distance (scalashogi Pos.dist).
inline int dist(int a, int b) {
  int dx = sqX(a) - sqX(b); if (dx < 0) dx = -dx;
  int dy = sqY(a) - sqY(b); if (dy < 0) dy = -dy;
  return dx > dy ? dx : dy;
}

// Piece id stored on the board: 0 = empty, otherwise (role << 1 | color) + 1.
inline uint8_t makePiece(Role r, Color c) {
  return static_cast<uint8_t>((static_cast<int>(r) << 1) | c) + 1;
}
inline Role roleOf(uint8_t p) { return static_cast<Role>((p - 1) >> 1); }
inline Color colorOf(uint8_t p) { return static_cast<Color>((p - 1) & 1); }

inline bool isLion(Role r) { return r == Lion || r == LionP; }
inline bool isRoyal(Role r) { return r == King || r == Prince; }
inline bool hasLionPower(Role r) {
  return r == Lion || r == LionP || r == Falcon || r == Eagle;
}

// Promotion table (Chushogi.promote in scalashogi). Returns -1 if the role
// does not promote.
inline int promoteRole(Role r) {
  switch (r) {
    case Pawn: return Tokin;
    case GoBetween: return ElephantP;
    case SideMover: return Boar;
    case VerticalMover: return Ox;
    case Rook: return DragonP;
    case Bishop: return HorseP;
    case Dragon: return Eagle;
    case Horse: return Falcon;
    case Elephant: return Prince;
    case Chariot: return Whale;
    case Tiger: return Stag;
    case Kirin: return LionP;
    case Phoenix: return QueenP;
    case Lance: return WhiteHorse;
    case Leopard: return BishopP;
    case Copper: return SideMoverP;
    case Silver: return VerticalMoverP;
    case Gold: return RookP;
    default: return -1;
  }
}

struct Dir { int8_t dx, dy; };

// Sente-POV direction atoms. Gote mirrors dy.
constexpr Dir kU{0, -1}, kD{0, 1}, kL{1, 0}, kR{-1, 0};
constexpr Dir kUL{1, -1}, kUR{-1, -1}, kDL{1, 1}, kDR{-1, 1};

struct RoleMoves {
  std::vector<Dir> direct;  // one-shot moves: steps and jumps
  std::vector<Dir> proj;    // sliding directions
};

// Movement tables for every role (Sente POV).
const std::array<RoleMoves, RoleCount>& roleMoves();

// A move inside the engine. mid == kNoSquare for ordinary moves; otherwise the
// intermediate square of a two-step lion-power move (orig, mid, dest in USI).
struct Move {
  int16_t from = kNoSquare;
  int16_t mid = kNoSquare;
  int16_t to = kNoSquare;
  bool promote = false;
};

}  // namespace chu
