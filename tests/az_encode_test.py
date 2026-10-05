#!/usr/bin/env python3
"""AZ encoding tests (milestone 1).

  1. Plane agreement: the C++ encoder (chushogi-az `encode`) and the Python
     mirror (az/encode.py) produce identical planes.
  2. Move mapping: every legal move (from chushogi-gen `legal`, flipped to
     the canonical frame for gote-to-move) maps to the same policy index in
     C++ (`legal-index`) and Python (move_to_index), and Python's
     index_to_move inverts it.

Positions: the initial position, all tsume records, and a stride sample of
real-game positions (including lion-capture third fields).

Usage: az_encode_test.py <games_dir1> [games_dir2 ...]
       [--az ./chushogi-az] [--gen ./chushogi-gen]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "az"))
import numpy as np

import encode as enc

INITIAL = ("lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/"
           "3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1")


class Cli:
    def __init__(self, binary):
        self.proc = subprocess.Popen([binary], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True,
                                     bufsize=1)

    def cmd(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()

    def set(self, sfen):
        assert self.cmd(f"position sfen {sfen}") == "ok"


def collect_positions(games_dirs, stride=7):
    yield INITIAL
    tsume = Path(__file__).parent / "tsume"
    for f in sorted(tsume.glob("*.json")):
        yield json.loads(f.read_text())["sfen"]
    for d in games_dirs:
        for sf in sorted(Path(d).glob("*/positions.sfen")):
            lines = sf.read_text().splitlines()
            for line in lines[::stride]:
                yield line.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games_dirs", type=Path, nargs="*")
    ap.add_argument("--az", default="./chushogi-az")
    ap.add_argument("--gen", default="./chushogi-gen")
    a = ap.parse_args()

    az, gen = Cli(a.az), Cli(a.gen)
    total = failures = 0
    for sfen in collect_positions(a.games_dirs):
        pos = enc.Position(sfen)
        az.set(sfen)
        gen.set(sfen)

        # 1. planes
        cpp = np.array(az.cmd("encode").split()[1:], dtype=np.float32)
        py = enc.encode_position(pos).reshape(-1)
        total += 1
        if cpp.shape != py.shape or not np.allclose(cpp, py, atol=1e-6):
            failures += 1
            diff = np.nonzero(~np.isclose(cpp, py, atol=1e-6))[0]
            print(f"FAIL planes {sfen}: {len(diff)} diffs, first {diff[:5]}")
            continue

        # 2. legal move indices
        legal_usi = gen.cmd("legal").split()[1:]
        flip = pos.side_to_move == 1
        want = sorted(enc.move_to_index(enc.flip_usi_move(m) if flip else m)
                      for m in legal_usi)
        got = sorted(int(i) for i in az.cmd("legal-index").split()[1:])
        total += 1
        if want != got:
            failures += 1
            missing = sorted(set(want) - set(got))
            extra = sorted(set(got) - set(want))
            print(f"FAIL indices {sfen}: missing={missing[:5]} "
                  f"extra={extra[:5]}")
            continue

        # 3. python round-trip of every legal move
        total += 1
        for m in legal_usi:
            cm = enc.flip_usi_move(m) if flip else m
            if enc.index_to_move(enc.move_to_index(cm)) != cm:
                failures += 1
                print(f"FAIL roundtrip {sfen}: {cm} -> "
                      f"{enc.move_to_index(cm)} -> "
                      f"{enc.index_to_move(enc.move_to_index(cm))}")
                break

    az.proc.stdin.write("quit\n")
    gen.proc.stdin.write("quit\n")
    print(f"{total - failures}/{total} az encode checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
