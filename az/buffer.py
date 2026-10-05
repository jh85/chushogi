"""Replay buffer for AZ self-play records.

Reads game JSON records produced by chushogi-az selfplay and serves shuffled
training batches: bit-packed planes (+ progress), sparse visit-count policy
targets, and the three value targets (z / rootQ for soft-Z / A0GB leaf).

Value-target selection lives here: --value-target z|softz|a0gb|blend with
blend y = (1-lambda) * z + lambda * rootQ  (lc0's q-blend recommendation).
"""

import base64
import json
from pathlib import Path

import numpy as np

import model as M


class Buffer:
    def __init__(self, games_dirs, max_positions=1_000_000, skip_plies=4):
        self.packed = []      # (1458,) uint8
        self.progress = []    # float16
        self.pol_idx = []     # list of int32 arrays
        self.pol_p = []       # list of float32 arrays
        self.z = []           # float32
        self.rootq = []
        self.a0gb = []
        files = []
        for d in games_dirs:
            files.extend(sorted(Path(d).rglob("game_*.json")))
        for f in files:
            try:
                rec = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue  # a game being written right now
            for smp in rec["samples"]:
                if smp["ply"] < skip_plies:
                    continue
                self.packed.append(
                    np.frombuffer(base64.b64decode(smp["planes"]),
                                  dtype=np.uint8).copy())
                self.progress.append(np.float16(smp["progress"]))
                idx = np.array([i for i, _ in smp["policy"]], dtype=np.int64)
                p = np.array([p for _, p in smp["policy"]],
                             dtype=np.float32)
                p = p / max(p.sum(), 1e-9)
                self.pol_idx.append(idx)
                self.pol_p.append(p)
                self.z.append(smp["z"])
                self.rootq.append(smp["rootq"])
                self.a0gb.append(smp["a0gb"])
        if len(self.packed) > max_positions:
            keep = slice(len(self.packed) - max_positions, len(self.packed))
            for lst in (self.packed, self.progress, self.pol_idx, self.pol_p,
                        self.z, self.rootq, self.a0gb):
                del lst[: len(lst) - max_positions]
        self.progress = np.asarray(self.progress, dtype=np.float16)
        self.z = np.asarray(self.z, dtype=np.float32)
        self.rootq = np.asarray(self.rootq, dtype=np.float32)
        self.a0gb = np.asarray(self.a0gb, dtype=np.float32)

    def __len__(self):
        return len(self.packed)

    def value_targets(self, mode, lam):
        """-> (N, 3) WDL-ish target distribution."""
        z, q, a = self.z, self.rootq, self.a0gb
        if mode == "z":
            v = z
        elif mode == "softz":
            v = q
        elif mode == "a0gb":
            v = a
        else:  # blend
            v = (1.0 - lam) * z + lam * q
        # scalar v (win prob, stm view) -> [win, draw, loss] target
        out = np.zeros((len(v), 3), dtype=np.float32)
        out[:, 0] = v
        out[:, 2] = 1.0 - v
        if mode == "z":
            out[z == 0.5] = [0.0, 1.0, 0.0]  # exact draw label
        elif mode == "blend":  # draws keep their draw mass, scaled by 1-lam
            d = z == 0.5
            out[d] = np.stack([(1.0 - lam) * 0.0 + lam * q[d],
                               np.full(d.sum(), 1.0 - lam),
                               (1.0 - lam) * 0.0 + lam * (1.0 - q[d])],
                              axis=1).astype(np.float32)
        return out

    def batch(self, idx):
        """-> planes tensor (B,82,12,12), sparse policy targets, wdl."""
        packed = np.stack([self.packed[i] for i in idx])
        prog = self.progress[idx].astype(np.float32)
        return packed, prog, ([self.pol_idx[i] for i in idx],
                              [self.pol_p[i] for i in idx])
