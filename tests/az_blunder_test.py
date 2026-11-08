#!/usr/bin/env python3
"""az_blunder_4j5i: the USI root royal filter.

Ply 319 of floodgate-600-10F+az_softz+az_a0gb+20261108123000: sente's only
royal is on 4j and in check. 14 of the 16 legal moves let gote capture it
next move (4j5i lost to +c 2i-5i); only 4j5j and 4j3j do not. The probes
mark those 14 children as proven losses, but every child is close to lost,
so visits follow the prior and the most-visited move was 4j5i. The filter
must skip every move that hangs the last royal. Here: USI mode, gate probe
defaults, random evaluator, one batch of 16 (every root move gets a visit,
so visits cannot tell the moves apart and the tie goes to the first legal
move, 4j4i, which hangs the king too), both colours (the gote case is the
same position mirrored, exercising the canonical flip).
"""

import re
import subprocess
import sys

SENTE = "4gtkt4/4se1h3f/f9m1/1c8o1/12/10s1/12/12/10+c1/2S5K3/12/2C5+r3 b - 319"
SAFE = {"4j5j", "4j3j"}


def mirror_sfen(sfen):
    board, side, hands, ply = sfen.split()
    rows = []
    for row in board.split("/"):
        cells = []
        for tok in re.findall(r"\d+|\+?[A-Za-z]", row):
            cells += [None] * int(tok) if tok.isdigit() else [tok.swapcase()]
        rows.append(cells[::-1])
    out = []
    for cells in rows[::-1]:
        s, run = "", 0
        for c in cells:
            if c is None:
                run += 1
                continue
            s += (str(run) if run else "") + c
            run = 0
        out.append(s + (str(run) if run else ""))
    return "/".join(out) + " " + ("w" if side == "b" else "b") + f" {hands} {ply}"


def mirror_move(m):
    sq = re.findall(r"(\d+)([a-l])", m)
    return "".join(f"{13 - int(f)}{chr(ord('a') + ord('l') - ord(r))}" for f, r in sq)


def main():
    az = subprocess.Popen(["./chushogi-az"], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True)

    def send(line):
        az.stdin.write(line + "\n")
        az.stdin.flush()

    def until(prefix):
        lines = []
        while True:
            line = az.stdout.readline()
            if not line:
                raise SystemExit("chushogi-az exited")
            lines.append(line.strip())
            if lines[-1].startswith(prefix):
                return lines

    send("usi")
    until("usiok")
    for opt in ("Eval value random", "Batch value 16"):
        send("setoption name " + opt)
    send("isready")
    until("readyok")

    total = failures = 0
    for sfen, safe in ((SENTE, SAFE),
                       (mirror_sfen(SENTE), {mirror_move(m) for m in SAFE})):
        for _ in range(3):
            send("usinewgame")
            send(f"position sfen {sfen}")
            send("go nodes 16")
            lines = until("bestmove")
            move = lines[-1].split()[1]
            total += 1
            if move not in safe:
                failures += 1
                print(f"FAIL az_blunder_4j5i: {sfen} -> {move}, want one of "
                      f"{sorted(safe)}")
                print("  " + "\n  ".join(lines))
    send("quit")
    az.wait()
    print(f"{total - failures}/{total} az blunder checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
