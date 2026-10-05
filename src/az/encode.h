// AlphaZero input encoding for chu shogi.
//
// Positions are canonicalized to the side-to-move's perspective first
// (Position::flipped() when gote is to move), then encoded as 82 planes of
// 12x12:
//
//   planes  0..38 : our pieces, one plane per Role
//   planes 39..77 : their pieces, one plane per Role
//   plane  78     : one-hot on lastLionCapture (lion-immunity game state)
//   plane  79     : broadcast 1 if the position occurred once before
//   plane  80     : broadcast 1 if it occurred twice or more before
//   plane  81     : broadcast ply / 1000 (1000-ply draw-cap progress)
//
// Policy space (50,688 logits): the core is the from x to matrix
// (144 x 144 = 20,736); promotions live in a second from x to matrix;
// two-step lion-power moves (lion / horned falcon / soaring eagle double
// moves, incl. igui) live in a from x midDir(8) x toDir(8) block.

#pragma once

#include <cstdint>
#include <vector>

#include "../position.h"

namespace az {

constexpr int kBoard = chu::kSquares;               // 144
constexpr int kPolicyMove = kBoard * kBoard;        // 20,736
constexpr int kPolicyPromote = kPolicyMove;         // second from-to matrix
constexpr int kPolicyDouble = kBoard * 8 * 8;       // two-step block
constexpr int kPolicySize = 2 * kPolicyMove + kPolicyDouble;  // 50,688

constexpr int kNumPlanes = 82;

// Move <-> policy index. Moves must be in the canonical (flipped) frame.
int moveToIndex(const chu::Move& m);
chu::Move indexToMove(int idx);
inline bool isDoubleIndex(int idx) { return idx >= 2 * kPolicyMove; }

// Encodes `pos` (already canonicalized by the caller) into `out`
// (kNumPlanes * kBoard floats, plane-major). repCount = how many times this
// position occurred earlier in the game (0/1/2+), ply = plies played.
void encodePlanes(const chu::Position& pos, int repCount, int ply,
                  float* out);

// Convenience: canonicalize + encode.
void encodePosition(const chu::Position& pos, int repCount, int ply,
                    float* out);

// Packs planes 0..80 (binary) MSB-first into out (81*144/8 = 1458 bytes);
// plane 81 (ply progress) is returned separately.
constexpr int kPackedBytes = 81 * kBoard / 8;
float packPlanes(const float* planes, uint8_t* out);

}  // namespace az
