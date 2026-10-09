// Conversion layer: SFEN <-> internal Position.
//
// Chu shogi SFEN as used by lishogi:
//   <board 12x12> <b|w> <last-lion-capture square or "-"> <move number>
// The third field normally holds hand pieces in standard shogi; chu shogi has
// no drops, so lishogi reuses it for the lion-trading rule state.
//
// Piece letters (uppercase = Sente, lowercase = Gote):
//   k king  q queen  n lion  r rook  b bishop  d dragon king  h dragon horse
//   g gold  s silver  c copper  e drunk elephant  f leopard  t blind tiger
//   o kirin  x phoenix  m side mover  v vertical mover  a reverse chariot
//   l lance  i go-between  p pawn
// Promoted pieces are prefixed with '+', keeping the base letter, e.g. +p is
// the tokin, +r the promoted rook (dragon king), +d the soaring eagle.

#pragma once

#include <string>

#include "position.h"

namespace chu {

// Parses a full SFEN ("board turn field3 movenum"). Missing trailing fields
// default to Sente to move, no lion capture, move number 1. The rule set is
// not part of the SFEN: out.rules is left as it was.
bool parseSfen(const std::string& sfen, Position& out);

std::string toSfen(const Position& pos);

}  // namespace chu
