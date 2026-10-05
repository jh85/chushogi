#include "encode.h"

#include <algorithm>

namespace az {

namespace {

// Fixed unit-direction table for the double-move block.
constexpr chu::Dir kDirs[8] = {{-1, -1}, {0, -1}, {1, -1}, {-1, 0},
                               {1, 0},  {-1, 1}, {0, 1},  {1, 1}};

int dirIndex(int dx, int dy) {
  for (int i = 0; i < 8; ++i)
    if (kDirs[i].dx == dx && kDirs[i].dy == dy) return i;
  return -1;
}

}  // namespace

int moveToIndex(const chu::Move& m) {
  if (m.mid == chu::kNoSquare) {
    const int idx = m.from * kBoard + m.to;
    return m.promote ? kPolicyMove + idx : idx;
  }
  const int md = dirIndex(chu::sqX(m.mid) - chu::sqX(m.from),
                          chu::sqY(m.mid) - chu::sqY(m.from));
  const int td = dirIndex(chu::sqX(m.to) - chu::sqX(m.mid),
                          chu::sqY(m.to) - chu::sqY(m.mid));
  return 2 * kPolicyMove + m.from * 64 + md * 8 + td;
}

chu::Move indexToMove(int idx) {
  chu::Move m;
  if (idx < 2 * kPolicyMove) {
    m.promote = idx >= kPolicyMove;
    const int i = idx % kPolicyMove;
    m.from = i / kBoard;
    m.to = i % kBoard;
    return m;
  }
  const int i = idx - 2 * kPolicyMove;
  m.from = i / 64;
  const int md = (i / 8) % 8, td = i % 8;
  const int fx = chu::sqX(m.from), fy = chu::sqY(m.from);
  m.mid = (fy + kDirs[md].dy) * chu::kFiles + fx + kDirs[md].dx;
  m.to = (chu::sqY(m.mid) + kDirs[td].dy) * chu::kFiles + chu::sqX(m.mid) +
         kDirs[td].dx;
  return m;
}

void encodePlanes(const chu::Position& pos, int repCount, int ply,
                  float* out) {
  std::fill(out, out + kNumPlanes * kBoard, 0.0f);
  for (int s = 0; s < kBoard; ++s) {
    const uint8_t p = pos.board[s];
    if (!p) continue;
    const int role = static_cast<int>(chu::roleOf(p));
    const bool ours = chu::colorOf(p) == pos.sideToMove;
    out[(ours ? role : 39 + role) * kBoard + s] = 1.0f;
  }
  if (pos.lastLionCapture != chu::kNoSquare)
    out[78 * kBoard + pos.lastLionCapture] = 1.0f;
  if (repCount >= 1) std::fill(out + 79 * kBoard, out + 80 * kBoard, 1.0f);
  if (repCount >= 2) std::fill(out + 80 * kBoard, out + 81 * kBoard, 1.0f);
  const float progress = std::min(ply / 1000.0f, 1.0f);
  std::fill(out + 81 * kBoard, out + 82 * kBoard, progress);
}

void encodePosition(const chu::Position& pos, int repCount, int ply,
                    float* out) {
  if (pos.sideToMove == chu::Gote)
    encodePlanes(pos.flipped(), repCount, ply, out);
  else
    encodePlanes(pos, repCount, ply, out);
}

}  // namespace az
