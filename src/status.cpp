#include "status.h"

#include "attacks.h"
#include "movegen.h"

namespace chu {

namespace {

// Dead pieces are pawns and lances already stuck on the last rank: they can
// never move again and count for neither side in the baring/material rules.
bool isDead(Role r, int sq, Color c) {
  return (r == Pawn || r == Lance) && sqY(sq) == backrankY(c);
}

// Chushogi.bareKing: was `color`'s king bared?
bool bareKing(const Position& pos, Color color) {
  int ourPieces = 0, ourRoyals = 0, ourRoyalSq = kNoSquare;
  int theirPieces = 0, theirRoyals = 0;
  bool adjacentThreat = false;
  int theirCounting[kSquares];
  int nTheirCounting = 0;

  for (int s = 0; s < kSquares; ++s) {
    const uint8_t p = pos.board[s];
    if (!p) continue;
    const Role r = roleOf(p);
    if (colorOf(p) == color) {
      if (!isDead(r, s, colorOf(p))) ++ourPieces;
      if (isRoyal(r)) {
        ++ourRoyals;
        ourRoyalSq = s;
      }
    } else {
      // For the winning side, pawns and go-betweens do not count (they must
      // first promote safely), and dead pieces count for nobody.
      const bool counts = r != Pawn && r != GoBetween &&
                          !isDead(r, s, colorOf(p));
      if (counts) {
        ++theirPieces;
        theirCounting[nTheirCounting++] = s;
      }
      if (isRoyal(r)) ++theirRoyals;
    }
  }
  // The adjacency test must not depend on the scan order: our royal may be
  // found after some of their counting pieces.
  if (ourRoyalSq != kNoSquare)
    for (int i = 0; i < nTheirCounting; ++i)
      if (dist(ourRoyalSq, theirCounting[i]) == 1) adjacentThreat = true;

  return ourPieces == 1 &&    // we have only a single (non-dead) piece
         ourRoyals == 1 &&    // and that piece is royal
         theirPieces > 1 &&   // opponent has more than just a single royal
         theirRoyals >= 1 &&  // but they have at least one royal
         !inCheck(pos, !color) &&  // no threat of immediate royal capture
         (theirPieces > 2 || !adjacentThreat);
  // opponent either has more pieces than we can capture back, or our lone
  // royal does not stand next to (threaten to bare) any of their pieces
}

// Chushogi.isInsufficientMaterial: only two non-dead pieces left, both royal,
// and neither side is in check.
bool insufficientMaterial(const Position& pos) {
  int live = 0, royalsSente = 0, royalsGote = 0;
  for (int s = 0; s < kSquares; ++s) {
    const uint8_t p = pos.board[s];
    if (!p) continue;
    const Role r = roleOf(p);
    if (!isDead(r, s, colorOf(p))) ++live;
    if (isRoyal(r)) (colorOf(p) == Sente ? royalsSente : royalsGote)++;
  }
  return live == 2 && royalsSente == 1 && royalsGote == 1 &&
         !inCheck(pos, Sente) && !inCheck(pos, Gote);
}

int royalCount(const Position& pos, Color color) {
  int n = 0;
  for (int s = 0; s < kSquares; ++s) {
    const uint8_t p = pos.board[s];
    if (p && colorOf(p) == color && isRoyal(roleOf(p))) ++n;
  }
  return n;
}

}  // namespace

StatusResult evaluateStatus(const Position& pos) {
  // Same order as scalashogi Chushogi.status (minus repetition).
  if (royalCount(pos, pos.sideToMove) == 0)
    return {GameEnd::RoyalsLost, !pos.sideToMove};
  const bool bareSente = bareKing(pos, Sente);
  const bool bareGote = bareKing(pos, Gote);
  if (bareSente || bareGote)
    return {GameEnd::BareKing, bareSente ? Gote : Sente};
  if (generateLegal(pos).empty())
    return {GameEnd::Stalemate, !pos.sideToMove};
  if (insufficientMaterial(pos)) return {GameEnd::Draw, -1};
  return {GameEnd::Playing, -1};
}

const char* gameEndName(GameEnd end) {
  switch (end) {
    case GameEnd::Playing: return "playing";
    case GameEnd::RoyalsLost: return "royalslost";
    case GameEnd::BareKing: return "bareking";
    case GameEnd::Stalemate: return "stalemate";
    case GameEnd::Draw: return "draw";
  }
  return "?";
}

const char* colorName(int color) {
  return color == Sente ? "sente" : color == Gote ? "gote" : "-";
}

}  // namespace chu
