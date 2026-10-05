#!/usr/bin/env python3
"""Head-to-head comparison of the mate solver's BNS and df-pn arithmetics
over the tsume records: same position, same node cap, compare verdict and
node count.

Usage: mate_compare.py <json_dir> [--binary ./chushogi-gen] [--jobs J]
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run_one(path, binary):
    rec = json.loads(path.read_text())
    if rec.get("conditional"):
        return None
    recorded = rec.get("engine_check", {}).get("mate", {})
    if recorded.get("result") not in ("yes", "no"):
        return None  # unknown/timeout verdicts are cap-bound, not informative
    base_nodes = recorded.get("nodes", 0)
    cap = max(300000, 2 * base_nodes + 1000)
    row = {"id": rec["id"], "expected": recorded.get("result")}
    for arith in ("bns", "pndn"):
        cmd = f"position sfen {rec['sfen']}\nmate {cap} {arith}\nquit\n"
        try:
            out = subprocess.run([binary], input=cmd, capture_output=True,
                                 text=True, timeout=600).stdout
        except subprocess.TimeoutExpired:
            row[arith] = ("timeout", 0)
            continue
        lines = out.splitlines()
        verdict = next((l.split()[1] for l in lines if l.startswith("mate ")),
                       "error")
        nodes = next((int(l.split()[1]) for l in lines
                      if l.startswith("nodes ")), 0)
        row[arith] = (verdict, nodes)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_dir", type=Path)
    ap.add_argument("--binary", default="./chushogi-gen")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    files = sorted(a.json_dir.glob("*.json"))
    rows = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for row in ex.map(lambda p: run_one(p, a.binary), files):
            if row:
                rows.append(row)
                print(json.dumps(row), flush=True)

    agree = sum(1 for r in rows if r["bns"][0] == r["pndn"][0])
    print(f"\nverdicts agree: {agree}/{len(rows)}")
    for r in rows:
        if r["bns"][0] != r["pndn"][0]:
            print("  disagree:", r["id"], r["bns"], r["pndn"])
    for verdict in ("yes", "no"):
        sub = [r for r in rows if r["expected"] == verdict
               and r["bns"][0] == verdict and r["pndn"][0] == verdict]
        if sub:
            bn = sum(r["bns"][1] for r in sub)
            pn = sum(r["pndn"][1] for r in sub)
            print(f"{verdict}: both proved {len(sub)}, nodes bns={bn} "
                  f"pndn={pn} ratio={pn / max(bn, 1):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
