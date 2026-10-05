"""AlphaZero network for chu shogi (PyTorch).

JHBR3-BT4-style transformer over the 144 board squares:
  - per-square input embedding from the 82 input planes
  - pre-norm transformer trunk
  - policy head: Q@K^T per-square pair projections -> the 144x144 from-to
    matrix (x2: normal + promotion), plus a per-square MLP producing the
    64 double-move (two-step lion-power) logits per from-square
  - value head: pooled -> 3-way WDL

Policy layout (50,688): [from-to 20,736 | from-to-promote 20,736 |
double-move 9,216], matching src/az/encode.cpp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD = 144
NUM_PLANES = 82
POLICY_MOVE = BOARD * BOARD
POLICY_DOUBLE = BOARD * 64
POLICY_SIZE = 2 * POLICY_MOVE + POLICY_DOUBLE


class Block(nn.Module):
    def __init__(self, d, heads, ffn_mult=4):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.RMSNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(),
                                 nn.Linear(ffn_mult * d, d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.n2(x))
        return x


class ChuNet(nn.Module):
    def __init__(self, d_model=256, blocks=6, heads=8):
        super().__init__()
        self.embed = nn.Linear(NUM_PLANES, d_model)
        self.pos = nn.Parameter(torch.randn(1, BOARD, d_model) * 0.02)
        self.trunk = nn.ModuleList(Block(d_model, heads)
                                   for _ in range(blocks))
        self.norm = nn.RMSNorm(d_model)
        self.q1 = nn.Linear(d_model, 64)
        self.k1 = nn.Linear(d_model, 64)
        self.q2 = nn.Linear(d_model, 64)
        self.k2 = nn.Linear(d_model, 64)
        self.dbl = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(),
                                 nn.Linear(128, 64))
        self.vpool = nn.Linear(d_model, 8)
        self.vfc = nn.Sequential(nn.Linear(8 * BOARD, 256), nn.GELU(),
                                 nn.Linear(256, 3))

    def forward(self, planes):
        """planes: (B, 82, 12, 12) -> (policy logits (B, 50688), wdl (B, 3))."""
        b = planes.shape[0]
        x = planes.reshape(b, NUM_PLANES, BOARD).permute(0, 2, 1)  # B,144,82
        x = self.embed(x) + self.pos
        for blk in self.trunk:
            x = blk(x)
        x = self.norm(x)
        m1 = torch.matmul(self.q1(x), self.k1(x).transpose(1, 2))  # B,144,144
        m2 = torch.matmul(self.q2(x), self.k2(x).transpose(1, 2))
        dbl = self.dbl(x)                                          # B,144,64
        policy = torch.cat([m1.reshape(b, -1), m2.reshape(b, -1),
                            dbl.reshape(b, -1)], dim=1)
        v = self.vpool(x).reshape(b, 8 * BOARD)
        wdl = self.vfc(v)
        return policy, wdl


def unpack_batch(raw, progress):
    """Bit-packed planes -> torch tensor (B, 82, 12, 12).

    raw: bytes, B * (81*144/8); progress: float array of B (ply/1000).
    """
    import numpy as np
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    b = bits.size // (81 * BOARD)
    planes = np.zeros((b, NUM_PLANES, 12, 12), dtype=np.float32)
    planes[:, :81] = bits.reshape(b, 81, 12, 12)
    planes[:, 81] = np.asarray(progress, dtype=np.float32)[:, None, None]
    return torch.from_numpy(planes)
