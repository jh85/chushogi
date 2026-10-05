#!/usr/bin/env python3
"""Rule unit tests for the chu shogi move generator, using crafted positions.

Each case gives an SFEN plus moves that must be present in / absent from the
generated legal move list. The cases target the tricky parts that game
replays cannot prove (over-generation): lion-trading restrictions, lion
two-step geometry, promotion conditions, and the lion-capture history field.

Usage: rule_test.py [--binary ./chushogi-gen]
"""

import argparse
import subprocess
import sys

INITIAL = ("lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/"
           "3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1")

# (name, sfen, present, absent)
CASES = [
    # --- Lion trading rules -------------------------------------------------
    # I. Adjacent lion capture is always allowed, even if the lion is
    #    protected (gold at 3a protects 2b).
    ("I.adjacent-lion-capture",
     "9g2/10n1/11N/12/12/12/12/12/12/12/12/12 b - 1",
     ["1c2b", "1c2b1c", "1c2b3c"],
     []),
    # II. Non-adjacent lion capture of a protected lion is forbidden
    #     (rook at 2a protects 2e along the file).
    ("II.protected-lion",
     "10r1/12/11N/12/10n1/12/12/12/12/12/12/12 b - 1",
     ["1c2d", "1c1d2d"],
     ["1c2e", "1c1d2e", "1c2d2e", "1c1b2e"]),
    # III. Same but unprotected: allowed.
    ("III.unprotected-lion",
     "12/12/11N/12/10n1/12/12/12/12/12/12/12 b - 1",
     ["1c2e", "1c1d2e"],
     []),
    # IV. Hidden protector: the bishop's diagonal to 1e is currently blocked
    #     by the capturing lion itself (X-ray); capture is still forbidden.
    ("IV.hidden-protector",
     "7b4/12/9N2/12/11n/12/12/12/12/12/12/12 b - 1",
     [],
     ["3c1e", "3c2d1e", "3c3d1e"]),
    ("IV.hidden-protector-control",
     "12/12/9N2/12/11n/12/12/12/12/12/12/12 b - 1",
     ["3c1e", "3c2d1e"],
     []),
    # V. Kuisoe: capturing something substantial (silver) alongside the lion
    #    makes the capture legal even though the lion is protected (rook 3a).
    ("V.kuisoe",
     "9r2/12/10N1/9s2/9n2/12/12/12/12/12/12/12 b - 1",
     ["2c3d3e"],
     ["2c3e", "2c2d3e"]),
    # VI. A pawn (or go-between) defender still counts even when captured on
    #     the way (Japanese Chu Shogi Association interpretation).
    ("VI.pawn-defender",
     "12/12/10N1/9p2/9n2/12/12/12/12/12/12/12 b - 1",
     ["2c3d"],
     ["2c3e", "2c3d3e", "2c2d3e"]),
    # VII. After a non-lion captured a lion, a non-lion may not capture a lion
    #      on a *different* square; the same square is allowed (e.g. a kirin
    #      that captured and promoted to lion may be recaptured).
    ("VII.lion-recapture-ban",
     "7N3r/12/7+O3r/12/12/12/12/12/12/12/12/12 w 5c 10",
     ["1c5c"],
     ["1a5a"]),
    ("VII.lion-recapture-ban-off",
     "7N3r/12/7+O3r/12/12/12/12/12/12/12/12/12 w - 10",
     ["1c5c", "1a5a"],
     []),
    # --- Lion move geometry -------------------------------------------------
    ("VIII.jitto-igui",
     "12/12/12/12/12/6Np4/12/12/12/12/12/12 b - 1",
     ["6f5f6f",  # igui: capture pawn on 5f without moving
      "6f6g6f",  # jitto: pass a turn via empty 6g
      "6f5f",    # plain capture
      "6f5f4f", "6f5f5g",  # capture and move on
      "6f7f"],
     []),
    ("IX.lion-jumps",
     "12/12/12/12/7P4/6NP4/6p5/12/12/12/12/12 b - 1",
     ["6f6h",     # jump over the enemy pawn without capturing it
      "6f6g6h",   # or capture it on the way
      "6f6g6f", "6f6g",
      "6f4f",     # jump over own pawn on 5f
      "6f4e"],    # knight-shaped jump
     ["6f5f", "6f5f6f"]),  # cannot step onto / capture own piece
    # --- Horned falcon / soaring eagle --------------------------------------
    ("X.eagle",
     "12/12/8+D3/12/12/12/12/12/12/12/12/12 b - 1",
     ["4c5b4c", "4c5b6a", "4c6a",      # forward-right lion power
      "4c3b4c", "4c3b2a", "4c2a",      # forward-left lion power
      "4c4b", "4c4a",                  # straight slide (rook-like)
      "4c7f", "4c1c", "4c3d"],         # backward diagonal / side slides
     ["4c1a",    # forward diagonal beyond two squares
      "4c2b",    # knight square: no
      "4c5b6a+",  # eagle cannot promote (already promoted)
      "4c3d4c"]),  # lion power only on the forward diagonals
    ("XI.falcon",
     "12/12/8+H3/12/12/12/12/12/12/12/12/12 b - 1",
     ["4c4b", "4c4a", "4c4b4c", "4c4b4a",  # forward lion power
      "4c3b", "4c2a", "4c5c", "4c4d"],    # bishop/side/back slides
     ["4c3c4c",   # no two-step sideways
      "4c4a4c",   # mid step must be adjacent
      "4c4d4c"]),  # no two-step backwards
    # --- Promotion conditions ------------------------------------------------
    ("XII.promotion",
     "12/5P6/12/3G2P5/3p3P2E1/1O10/2G9/2p9/12/12/12/12 b - 1",
     ["5e5d", "5e5d+",      # pawn entering the zone
      "6d6c",               # pawn inside the zone: no promotion
      "7b7a", "7b7a+",      # pawn reaching the last rank inside the zone
      "9d9e", "9d9e+",      # capture while starting inside the zone
      "11f11d", "11f11d+",  # kirin jumping into the zone
      "2e2d", "2e2d+",      # elephant entering the zone (-> prince)
      "10g10h"],            # capture entirely outside the zone: no promotion
     ["6d6c+", "10g10h+"]),
    ("XIII.promotion-gote",
     "12/12/12/12/12/12/12/7p4/12/7p4/6p5/12 w - 1",
     ["5h5i", "5h5i+", "6k6l", "6k6l+", "5j5k"],
     ["5j5k+"]),
    # --- Immunity from capture (lion trading rules), with kings -------------
    # R1: lion capturing lion. Adjacent captures are always allowed; a
    # non-adjacent capture is forbidden if the lion could be recaptured on
    # the next move, unless something substantial (not pawn/go-between) is
    # captured alongside (tsukegui/kuisoe).
    ("R1.adjacent-capture-of-protected-lion",
     "k11/12/12/12/7g4/6n5/5N6/12/12/12/12/11K b - 1",
     ["7g6f",     # LnxLn: adjacent, so the gold's protection is irrelevant
      "7g6f5e",   # ...and the lion may even take the gold afterwards
      "7g6f6e"],  # ...or end on a square the gold attacks
     []),
    ("R1.nonadjacent-protected-lion",
     "k11/12/12/12/4B2l4/4S7/5N6/7n4/12/12/12/11K b - 1",
     ["8e5h",     # BxLn: rule 1 only restricts LION capturing lion
      "7g5e"],    # lion may capture the protecting lance itself
     ["7g5h",     # LnxLn forbidden: white lance protects along the file
      "7g5f5h", "7g6g5h"]),  # two-step paths are equally forbidden
    ("R1.nonadjacent-protected-lion-w",
     "k11/12/12/12/4B2l4/4S7/5N6/7n4/12/12/12/11K w - 1",
     ["5h5j", "5h6g", "5h4g5h"],
     ["5h7g",     # black lion is protected by the silver
      "5h6g7g", "5h4g7g"]),
    ("R1.hidden-protector",
     "k11/12/12/12/3n8/12/5N6/5P6/7b4/12/12/11K b - 1",
     ["7g5i",     # the lion may capture the hidden bishop itself
      "7g8f"],
     ["7g9e",     # X-ray: bishop 5i-6h-7g-8f-9e is unblocked by moving
      "7g8e9e", "7g8f9e", "7g6f9e"]),
    ("R1.hidden-protector-w",
     "k11/12/12/12/3n8/12/5N6/5P6/7b4/12/12/11K w - 1",
     ["5i7g",     # BxLn is legal (non-lion capture, no ban active)
      "9e10e"],
     ["9e7g",     # black pawn on 7h protects the black lion
      "9e8f7g"]),
    ("R1.tsukegui",
     "k11/12/12/12/6+o1r3/4gi6/6N5/7s4/8n3/12/12/11K b - 1",
     ["6g5h4i",   # capture the silver AND the lion: kuisoe, always legal
      "6g7f",     # capturing the go-between alone is fine
      "6g7f8f"],  # go-between + gold: not a lion capture, fine
     ["6g4i",     # corner lion is protected by the rook
      "6g6e",     # this lion is protected by the rook too
      "6g7f6e"]), # go-between is not a substantial capture alongside
    ("R1.pawn-defender",
     "k11/12/12/12/12/4N7/4p7/4n7/12/12/12/11K b - 1",
     ["8f8g"],    # taking the pawn alone is fine
     ["8f8h",     # lion protected by the pawn: forbidden
      "8f8g8h"]), # JCSA: the pawn still counts even captured on the way
    ("R1.gobetween-defender",
     "k11/12/12/12/12/4N7/4i7/4n7/12/12/12/11K b - 1",
     ["8f8g"],
     ["8f8h", "8f8g8h"]),  # same with a go-between defender
    ("R1.multi-lion-exposure",
     "k11/12/3l8/12/3b4+o3/6n5/4N7/6R5/7K4/12/12/12 b - 1",
     ["8g8f"],
     ["8g6f"]),   # the white lion on 4e would recapture within distance 2
    # R2: after a non-lion captured a lion, a non-lion may not capture a lion
    # on a DIFFERENT square on the next move. The ban is encoded in the third
    # SFEN field. Lishogi implements the strict Edo-era rule: NO Okazaki
    # amendment (an unprotected lion may still not be captured).
    ("R2.basic-ban-and-okazaki",
     "k11/12/11n/12/4r7/12/12/11R/12/12/12/11K b 5e 2",
     [],
     ["1h1c"]),   # RxLn of a completely UNPROTECTED lion: still banned
    ("R2.basic-ban-off",
     "k11/12/11n/12/4r7/12/12/11R/12/12/12/11K b - 2",
     ["1h1c"],
     []),
    ("R2.same-square-exception",
     "k11/12/12/12/12/12/12/12/9n1n/4+o6P/12/6B4K b 8j 2",
     ["6l8j"],    # a kirin that captured and promoted to lion ON 8j may be
                  # recaptured on 8j ("another square" stipulation)
     ["6l3i",     # ...but not the lion on 3i
      "1j1i"]),   # ...and not the lion on 1i either
    ("R2.same-square-exception-off",
     "k11/12/12/12/12/12/12/12/9n1n/4+o6P/12/6B4K b - 2",
     ["6l8j", "6l3i", "1j1i"],
     []),
    ("R2.hit-and-run-falcon",
     "k11/12/12/12/6P5/12/12/5+h6/12/5N6/12/11K w 6e 2",
     ["7h7i", "7h7i7h"],
     ["7h7j",     # single jump capture of the lion: banned
      "7h7i7j"]), # second-step lion capture: banned
    ("R2.igui-midstep-quirk-falcon",
     "k11/12/12/12/6P5/12/12/5+h6/5N6/12/12/11K w 6e 2",
     ["7h7i7h",   # igui capture of the adjacent lion at the MID step: ALLOWED
      "7h7i7j"],  # capture lion at mid step and continue forward: ALLOWED
     ["7h7i"]),   # ...while the plain single-step capture is banned!
    ("R2.hit-and-run-eagle",
     "k11/12/12/12/6P5/12/12/7+d4/12/5N6/12/11K w 6e 2",
     ["5h6i", "5h6i5h"],
     ["5h7j",     # forward-diagonal jump capture of the lion: banned
      "5h6i7j"]), # second-step lion capture: banned
    ("R2.igui-midstep-quirk-eagle",
     "k11/12/12/12/6P5/12/12/7+d4/6N5/12/12/11K w 6e 2",
     ["5h6i5h",   # igui capture of the adjacent lion at the mid step: ALLOWED
      "5h6i7j"],  # capture lion at mid step and continue: ALLOWED
     ["5h6i"]),   # ...while the plain single-step capture is banned!
    ("R2.kirin-recapture-no-promo",
     "k11/12/12/12/12/12/12/7+d4/12/5O6/12/11K w 7j 2",
     ["5h7j",     # the kirin on 7j is not a lion: capturable (also same square)
      "5h6i7j", "5h6i"],
     []),
    ("R2.lion-immune-to-ban",
     "k11/12/12/12/7r4/3r4N3/12/3n4n3/12/12/12/11K b 5e 2",
     ["4f4h",     # the ban never restricts lions themselves
      "4f4g4h"],
     []),
    ("R2.king-cannot-take-lion",
     "k11/12/12/12/4r7/6n5/6K5/12/12/12/12/12 b 5e 2",
     [],
     ["6g6f"]),   # the king is a non-lion piece: banned as well
    ("R2.king-cannot-take-lion-off",
     "k11/12/12/12/4r7/6n5/6K5/12/12/12/12/12 b - 2",
     ["6g6f"],
     []),
    # --- Misc ----------------------------------------------------------------
    # XV. Last-rank promotion exception: lishogi grants it to BOTH pawns and
    #     lances (scalashogi: List(Pawn, Lance)); go-betweens are excluded
    #     because they can move backwards. All three pieces are already
    #     inside the zone here (rank b), moving quietly to the last rank.
    ("XV.last-rank-lance-exception",
     "12/5P1I1L2/12/12/12/12/12/12/12/12/12/12 b - 1",
     ["3b3a", "3b3a+",   # lance inside the zone: MAY promote on the last rank
      "7b7a", "7b7a+",   # pawn likewise
      "5b5a"],           # go-between may move there...
     ["5b5a+"]),         # ...but may NOT promote (no exception for it)
    ("XVI.last-rank-lance-exception-gote",
     "12/12/12/12/12/12/12/12/12/12/7l4/12 w - 1",
     ["5k5l", "5k5l+"],
     []),
    # --- Pawn/lance reaching the last rank: promote or defer ----------------
    # XVII. Sente. Pawn 2b and lance 4b are already inside the zone; lance 6e
    #     and pawn 8e enter from outside; go-betweens 10b (inside) and 12e
    #     (outside). Reaching rank a is always promote-or-defer for pawns and
    #     lances; the inside-zone go-between may not promote on the last rank.
    ("XVII.last-rank-sente",
     "12/2I5L1P1/12/12/I3P1L5/12/12/12/12/12/12/12 b - 1",
     ["2b2a", "2b2a+",          # pawn, inside zone -> last rank
      "4b4a", "4b4a+",          # lance, inside zone -> last rank
      "6e6d", "6e6d+", "6e6c", "6e6c+", "6e6b", "6e6b+", "6e6a", "6e6a+",
                              # lance entering the zone: promote on ANY step
      "8e8d", "8e8d+",          # pawn entering the zone
      "10b10a",                 # go-between inside zone: defer only
      "12e12d", "12e12d+"],     # go-between entering the zone: may promote
     ["10b10a+", "12e12a", "12e12a+"]),
    # XVIII. Same, mirrored, for gote (zone = ranks i-l, last rank = l).
    ("XVIII.last-rank-gote",
     "12/12/12/12/12/12/12/i3p1l5/12/12/2i5l1p1/12 w - 1",
     ["2k2l", "2k2l+", "4k4l", "4k4l+",
      "6h6i", "6h6i+", "6h6j", "6h6j+", "6h6k", "6h6k+", "6h6l", "6h6l+",
      "8h8i", "8h8i+", "10k10l", "12h12i", "12h12i+"],
     ["10k10l+", "12h12l", "12h12l+"]),
    # XIX. Captures on the last rank: any piece (even a go-between) may
    #     promote when it captures inside the zone; a quiet move inside the
    #     zone that does not reach the last rank may not promote.
    ("XIX.last-rank-captures",
     "4g1r1b1g1/6I3P1/4L7/12/8L3/12/12/12/12/12/12/12 b - 1",
     ["2b2a", "2b2a+",   # pawn captures gold on 2a
      "4e4a", "4e4a+",   # lance captures bishop on 4a (also entering)
      "6b6a", "6b6a+",   # go-between captures rook on 6a: capture -> promote
      "8c8a", "8c8a+",   # lance captures gold on 8a from within the zone
      "8c8b"],           # lance moving quietly within the zone
     ["8c8b+",           # ...may not promote (no capture, not last rank)
      "6b6b+"]),         # nonsense
    # XX/XXI. Deferred pieces on the last rank: an unpromoted pawn or lance is
    #     dead (no moves at all); a go-between can still step back, but that
    #     quiet zone-exit does not promote. Exact move count asserted.
    ("XX.dead-pieces-sente",
     "5I1L1P2/12/12/12/12/12/12/12/12/12/12/12 b - 1",
     ["7a7b"],
     ["7a7b+", "3a3b", "5a5b"],
     1),
    ("XXI.dead-pieces-gote",
     "12/12/12/12/12/12/12/12/12/12/12/5i1l1p2 w - 1",
     ["7l7k"],
     ["7l7k+", "3l3k", "5l5k"],
     1),
    ("XIV.initial-position",
     INITIAL,
     ["8i8h", "4h4g", "9h9g",     # pawn and go-between pushes
      "7j5h", "7j6h", "7j7h", "7j8h", "7j9k",  # lion jumps out of the cage
      "7k9k", "11j11k", "9j9k"],
     ["7j7i7h",  # own pawn on the mid square
      "4i4h", "9i9h",  # pawns blocked by own go-betweens
      "6l6k",    # own phoenix
      "7l7k",    # king onto own kirin
      "1l1k"]),  # own reverse chariot
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./chushogi-gen")
    a = ap.parse_args()

    class Engine:
        def __init__(self, binary):
            self.binary = binary
            self.start()

        def start(self):
            self.proc = subprocess.Popen(
                [self.binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1)

        def query(self, line):
            try:
                self.proc.stdin.write(line + "\n")
                self.proc.stdin.flush()
                out = self.proc.stdout.readline()
                if not out:
                    raise BrokenPipeError
                return out.rstrip("\n")
            except (BrokenPipeError, ValueError):
                self.start()  # engine exited (e.g. rejected position): respawn
                raise RuntimeError("engine died")

        def close(self):
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.wait()

    eng = Engine(a.binary)
    failures = 0
    for case in CASES:
        name, sfen, present, absent = case[:4]
        expected_count = case[4] if len(case) > 4 else None
        try:
            if eng.query(f"position sfen {sfen}") != "ok":
                print(f"FAIL {name}: position rejected")
                failures += 1
                continue
            out = eng.query("legal")
            moves = set(out.split()[1:])
            echo = eng.query("sfen")
        except RuntimeError:
            print(f"FAIL {name}: engine rejected the position")
            failures += 1
            continue
        missing = [m for m in present if m not in moves]
        extra = [m for m in absent if m in moves]
        count_bad = expected_count is not None and len(moves) != expected_count
        # SFEN round-trip must be exact.
        roundtrip_ok = echo == f"sfen {sfen}"
        if missing or extra or not roundtrip_ok or count_bad:
            failures += 1
            print(f"FAIL {name}")
            if missing:
                print(f"  missing legal moves: {missing}")
            if extra:
                print(f"  illegally generated: {extra}")
            if not roundtrip_ok:
                print(f"  sfen roundtrip: got {echo!r}, want {'sfen ' + sfen!r}")
            if count_bad:
                print(f"  expected {expected_count} legal moves, got {len(moves)}: {sorted(moves)}")
        else:
            print(f"PASS {name} ({len(moves)} legal moves)")
    eng.close()

    print(f"\n{len(CASES) - failures}/{len(CASES)} rule cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
