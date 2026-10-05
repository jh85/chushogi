#!/usr/bin/env python3
"""Run the engine's mate solver over extracted tsume JSON records and
update each record's engine_check.mate field in place.

Usage: mate_batch.py <json_dir> [--binary ./chushogi-gen] [--nodes N]
       [--jobs J] [--timeout S]
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def solve(path, binary, nodes, timeout):
    rec = json.loads(path.read_text())
    if rec.get("conditional"):
        return path.name, "skip", 0
    cmd = f"position sfen {rec['sfen']}\nmate {nodes}\nquit\n"
    try:
        out = subprocess.run([binary], input=cmd, capture_output=True,
                             text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        rec.setdefault("engine_check", {})["mate"] = {"result": "timeout"}
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
        return path.name, "timeout", 0
    lines = out.splitlines()
    verdict = next((l.split()[1] for l in lines if l.startswith("mate ")),
                   "error")
    nodes_used = next((int(l.split()[1]) for l in lines
                       if l.startswith("nodes ")), 0)
    pv = next((l.split()[1:] for l in lines if l.startswith("pv ")), [])
    rec.setdefault("engine_check", {})["mate"] = {
        "result": verdict, "nodes": nodes_used, "pv": pv}
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
    return path.name, verdict, nodes_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_dir", type=Path)
    ap.add_argument("--binary", default="./chushogi-gen")
    ap.add_argument("--nodes", type=int, default=3000000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=240)
    a = ap.parse_args()

    files = sorted(a.json_dir.glob("*.json"))
    counts = {}
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for name, verdict, nodes in ex.map(
                lambda p: solve(p, a.binary, a.nodes, a.timeout), files):
            counts[verdict] = counts.get(verdict, 0) + 1
            if verdict not in ("yes", "skip"):
                print(f"{name}: {verdict} ({nodes} nodes)", flush=True)
    print("summary:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
