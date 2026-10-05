#!/usr/bin/env python3
"""AZ self-play smoke test (milestone 2).

Plays two games with the random evaluator and validates the records:
  - JSON shape; policy probs sum to 1, indices inside the 50,688 space
  - moves replay legally through the main engine; the recorded terminal
    reason/result matches the engine's own status adjudication
  - packed planes decode (numpy.unpackbits) to exactly what the Python
    encoder produces from the independently replayed position

Usage: az_selfplay_test.py [--az ./chushogi-az] [--gen ./chushogi-gen]
"""

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "az"))
import numpy as np

import encode as enc


class Cli:
    def __init__(self, binary):
        self.proc = subprocess.Popen([binary], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True,
                                     bufsize=1)

    def cmd(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()


def unpack_planes(b64, progress):
    """packed 81 bit planes + progress scalar -> (82, 144) float32."""
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    bits = np.unpackbits(raw).reshape(81, enc.BOARD).astype(np.float32)
    out = np.zeros((enc.NUM_PLANES, enc.BOARD), dtype=np.float32)
    out[:81] = bits
    out[81, :] = progress
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--az", default="./chushogi-az")
    ap.add_argument("--gen", default="./chushogi-gen")
    a = ap.parse_args()

    az, gen = Cli(a.az), Cli(a.gen)
    tmp = Path(tempfile.mkdtemp(prefix="azsp_"))
    games = 2
    first = az.cmd(f"selfplay --games {games} --sims 16 --batch 4 --seed 7 "
                   f"--out {tmp}")
    assert first.startswith("game 0"), first
    for _ in range(games - 1):  # drain the remaining per-game lines
        az.proc.stdout.readline()

    total = failures = 0
    for path in sorted(tmp.glob("game_*.json")):
        rec = json.loads(path.read_text())
        moves = rec["moves"]
        initial = rec["initial"]

        # moves replay legally through the main engine
        total += 1
        tail = " moves " + " ".join(moves) if moves else ""
        if gen.cmd(f"position sfen {initial}{tail}") != "ok":
            failures += 1
            print(f"FAIL {path.name}: moves not accepted")
            continue

        # terminal status agrees with the engine
        total += 1
        st = gen.cmd("status")
        if rec["reason"] in ("fourfold", "plycap"):
            ok = st == "status playing" and rec["result"] == 0
        else:
            want = {"1": "sente", "-1": "gote", "0": None}[str(rec["result"])]
            ok = st.startswith(f"status {rec['reason']}") and (
                want is None or st.endswith(f" {want}"))
        if not ok:
            failures += 1
            print(f"FAIL {path.name}: record {rec['reason']}/{rec['result']} "
                  f"vs engine {st}")

        for smp in rec["samples"][:: max(1, len(rec["samples"]) // 12)]:
            total += 1
            probs = [p for _, p in smp["policy"]]
            idxs = [i for i, _ in smp["policy"]]
            if not (abs(sum(probs) - 1.0) < 1e-4 and
                    all(0 <= i < enc.POLICY_SIZE for i in idxs) and
                    smp["z"] in (0.0, 0.5, 1.0) and
                    0.0 <= smp["rootq"] <= 1.0 and 0.0 <= smp["a0gb"] <= 1.0):
                failures += 1
                print(f"FAIL {path.name} ply {smp['ply']}: bad targets")
                continue
            # planes vs independent replay
            total += 1
            prefix = moves[:smp["ply"]]
            tail = " moves " + " ".join(prefix) if prefix else ""
            az.cmd(f"position sfen {initial}{tail}")
            sfen_k = az.cmd("sfen")[5:]
            want = enc.encode_position(enc.Position(sfen_k))
            got = unpack_planes(smp["planes"], smp["progress"])
            if not np.allclose(want.reshape(-1), got.reshape(-1),
                               atol=1e-6):
                failures += 1
                print(f"FAIL {path.name} ply {smp['ply']}: plane mismatch")
                nz = np.nonzero(want.reshape(-1) != got.reshape(-1))[0]
                print("  first diffs:", nz[:8])

    print(f"{total - failures}/{total} az selfplay checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
