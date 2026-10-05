#!/usr/bin/env python3
"""Extract tsume chu shogi problems from chushogi-renmei.com / asahi-net
DHTML problem pages into JSON records (SFEN position + USI solutions).

Two page families are supported:

1. dhtmlcb2.js pages (tumeshogi2/tukurimono*.html): board in `shoki[x][y]`
   arrays; moves in `te[1]` (main), `te[2]`... (variations starting at their
   branch point), encoded 0xFFT T with each byte = (x<<4 | y) + 0x11, OR-ed
   with KC_TORU['code'] capture markers and KC_NARU for promotion.

2. chush2.js pages (tumeshogi/kinu*.html, asahi-net tumechu*.html): same
   shoki[] board but different romanization (zou/dou/hyou/houou/hon_ou...);
   solutions in `sk[N]` strings: [i|n]FFTT[N][captured-code] where FF/TT are
   1-based zero-padded (x+1)(y+1) pairs, 'N' after the digits = promotion,
   and a lion double move is an 'i' entry (first step) immediately followed
   by an 'n' entry (second step). Multiple full-length solution lines are
   listed in `skSentaku = 'start, start, ...'`; each runs until '' or
   'OWARI'. `skmsg[N]` holds per-move comments.

Both: shoki is column-major (x = file - 1, y = 0 on gote's back rank = SFEN
rank a), leading '_' = gote piece, a "2" suffix marks promoted-form names.

With --binary, records are validated against the engine: SFEN round-trip,
ply-by-ply legality of every line, and the BNS mate solver on the problem.

Usage:
  extract_tsume.py URL_OR_FILE [--id NAME] [--binary ./chushogi-gen]
                   [--mate-nodes N] [--outdir DIR]
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.request

# Page piece code -> lishogi SFEN letter ("+" prefix = promoted form).
# Covers both romanizations (dhtmlcb2.js and chush2.js).
PIECE = {
    'oosan': 'k', 'ousan': 'k', 'gyoku': 'k',
    'shishi': 'n', 'hon_oo': 'q', 'hon_ou': 'q', 'zoo': 'e', 'zou': 'e',
    'kin': 'g', 'gin': 's', 'doo': 'c', 'dou': 'c',
    'hyoo': 'f', 'hyou': 'f', 'yari': 'l', 'kirin': 'o',
    'hoo_oo': 'x', 'houou': 'x', 'tora': 't', 'kaku': 'b', 'hensha': 'a',
    'ryuu': 'd', 'mma': 'h', 'hisha': 'r', 'tate': 'v', 'yoko': 'm',
    'hyoko': 'p', 'chuunin': 'i',
    'shishi2': '+o', 'hon_oo2': '+x', 'hon_ou2': '+x', 'taishi2': '+e',
    'kimbisha2': '+g', 'tate2': '+s', 'yoko2': '+c', 'choro2': '+f',
    'hakku2': '+l', 'shika2': '+t', 'mma2': '+b', 'kujira2': '+a',
    'washi2': '+d', 'taka2': '+h', 'ryuu2': '+r', 'ushi2': '+v',
    'ino2': '+m', 'tokin2': '+p', 'zoo2': '+i', 'zou2': '+i',
}

RANKS = 'abcdefghijkl'


def square(x, y):
    return f"{x + 1}{RANKS[y]}"


def decode_page(raw_bytes):
    m = re.search(rb'charset=["\']?([\w-]+)', raw_bytes[:600], re.I)
    cs = m.group(1).decode().lower() if m else None
    if cs:
        enc = 'shift_jis' if ('shift' in cs or 'sjis' in cs) else 'utf-8'
        return raw_bytes.decode(enc, errors='replace')
    for enc in ('utf-8', 'shift_jis'):  # bare kifu .txt files carry no meta
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode('utf-8', errors='replace')


def clean_text(s):
    s = s.replace('\\/', '/').replace("\\'", "'").replace('\\n', ' ')
    s = re.sub(r"<[^>]*>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_board(txt):
    board = {}
    for m in re.finditer(r"shoki\[(\d+)\]\s*=\s*new Array\((.*?)\);", txt,
                         re.S):
        x = int(m.group(1))
        cells = re.findall(r"'([^']*)'", m.group(2))
        if len(cells) != 12:
            raise ValueError(f"shoki[{x}] has {len(cells)} cells")
        for y, code in enumerate(cells):
            if code == 'aki':
                continue
            gote = code.startswith('_')
            letter = PIECE[code.lstrip('_')]
            if not gote:
                letter = letter.upper() if letter[0] != '+' \
                    else '+' + letter[1].upper()
            board[(x, y)] = letter
    return board


def parse_te_lines(txt):
    """dhtmlcb2.js hex format -> {te_id: (combined USI moves, end_raw)}.

    A lion double move is a KC_IPPOME entry (first step) immediately
    followed by a KC_NIHOME entry (second step); they combine into one
    3-square USI move. end_raw[i] is the raw 1-based te index of the last
    entry of combined move i (bunki/bun indices are raw).
    """
    lines = {}
    for m in re.finditer(r"te\[(\d+)\]\s*=\s*\[0,(.*?)\n0\s*\];", txt, re.S):
        steps = []
        for line in m.group(2).split('\n'):
            mm = re.search(r"0x([0-9a-fA-F]+)", line)
            if not mm:
                continue
            v = int(mm.group(1), 16) & 0xffff
            f, t = (v >> 8) - 0x11, (v & 0xff) - 0x11
            steps.append(((square(f >> 4, f & 0xf), square(t >> 4, t & 0xf),
                           'KC_NARU' in line),
                          'KC_IPPOME' in line, 'KC_NIHOME' in line))
        moves, end_raw = [], []
        i = 0
        while i < len(steps):
            (fsq, tsq, naru), first, second = steps[i]
            if first:
                (_, tsq2, naru2), _, second2 = steps[i + 1]
                if not second2:
                    raise ValueError("KC_IPPOME not followed by KC_NIHOME")
                moves.append(fsq + tsq + tsq2 + ('+' if naru2 else ''))
                end_raw.append(i + 2)  # spans raw entries i+1, i+2
                i += 2
            elif second:
                raise ValueError("KC_NIHOME without KC_IPPOME")
            else:
                moves.append(fsq + tsq + ('+' if naru else ''))
                end_raw.append(i + 1)
                i += 1
        lines[int(m.group(1))] = (moves, end_raw)
    return lines


SK_MOVE = re.compile(r"^([in])?(\d{8})(N?)([a-z_0-9]*)$")


def sk_to_usi(raw_moves):
    """chush2.js string format -> USI moves (i/n pairs become one move)."""
    out = []
    i = 0
    while i < len(raw_moves):
        m = SK_MOVE.match(raw_moves[i])
        if not m:
            raise ValueError(f"unparseable sk move: {raw_moves[i]!r}")
        prefix, digits, naru, _cap = m.groups()
        fx, fy = int(digits[0:2]) - 1, int(digits[2:4]) - 1
        tx, ty = int(digits[4:6]) - 1, int(digits[6:8]) - 1
        if prefix == 'i':  # lion double move, first step; second follows
            m2 = SK_MOVE.match(raw_moves[i + 1])
            if not m2 or m2.group(1) != 'n':
                raise ValueError(f"'i' step not followed by 'n': "
                                 f"{raw_moves[i:i+2]!r}")
            d2 = m2.group(2)
            t2x, t2y = int(d2[4:6]) - 1, int(d2[6:8]) - 1
            out.append(square(fx, fy) + square(tx, ty) + square(t2x, t2y))
            i += 2
        else:
            out.append(square(fx, fy) + square(tx, ty) + ('+' if naru else ''))
            i += 1
    return out


def parse_sk_lines(txt):
    """chush2.js format -> (main, [variations as (branch, moves)])."""
    sk = {}
    for m in re.finditer(r"sk\[(\d+)\]\s*=\s*'([^']*)'", txt):
        sk[int(m.group(1))] = m.group(2)
    skmsg = {}
    for m in re.finditer(r"skmsg\[(\d+)\]\s*=\s*'(.*?)';", txt, re.S):
        skmsg[int(m.group(1))] = clean_text(m.group(2))
    m = re.search(r"skSentaku\s*=\s*'([^']*)'", txt)
    starts = [int(s) for s in m.group(1).split(',')] if m else [1]
    lines = []
    for start in starts:
        raw_moves, cms = [], []
        n = start
        while sk.get(n, '') not in ('', 'OWARI'):
            raw_moves.append(sk[n])
            cms.append(skmsg.get(n, ''))
            n += 1
        moves = sk_to_usi(raw_moves)
        # attach comments: an i/n pair shares one USI move
        aligned, ci = [], 0
        for r, c in zip(raw_moves, cms):
            if r.startswith('i'):
                c2 = cms[ci + 1] if ci + 1 < len(cms) else ''
                aligned.append(' / '.join(x for x in (c, c2) if x))
                ci += 2
            else:
                aligned.append(c)
                ci += 1
        lines.append((moves, aligned))
    return lines


def parse_page(txt):
    m = re.search(r"title\s*=\s*'(.*?)';", txt, re.S)
    title = clean_text(m.group(1)) if m else ''
    board = parse_board(txt)
    if re.search(r"sk\[1\]\s*=", txt):
        return title, board, 'sk', parse_sk_lines(txt)
    te = parse_te_lines(txt)
    if te:
        bun = {}
        for m in re.finditer(r"bun\[(\d+)\]\[(\d+)\]\s*=\s*'(.*?)';", txt,
                             re.S):
            bun[(int(m.group(1)), int(m.group(2)))] = clean_text(m.group(3))
        # bunki[N][M] = [0, teId, startPly, ...]: after line N's raw entry M,
        # the next move is either line N's own next entry or te[teId]'s raw
        # entry startPly. M and startPly are RAW indices: the two entries of
        # a lion double move count separately.
        bunki = {}
        for m in re.finditer(r"bunki\[(\d+)\]\[(\d+)\]\s*=\s*\[([0-9,\s]*)\];",
                             txt):
            vals = [int(v) for v in m.group(3).split(',') if v.strip()]
            bunki[(int(m.group(1)), int(m.group(2)))] = vals[1:]

        def prefix_len(te_id, raw_m):  # combined moves fully inside raw<=M
            return sum(1 for e in te[te_id][1] if e <= raw_m)

        def tail_from(te_id, raw_p):  # combined moves covering raw>=P
            moves, end_raw = te[te_id]
            for i, e in enumerate(end_raw):
                if e >= raw_p:
                    return moves[i:]
            return []

        def raw_comments(te_id):
            moves, end_raw = te[te_id]
            start_raw = [1] + [e + 1 for e in end_raw[:-1]]
            out = {}
            for (n, p), t in bun.items():
                if n != te_id:
                    continue
                for i, (s, e) in enumerate(zip(start_raw, end_raw)):
                    if s <= p <= e:
                        out[str(i + 1)] = t
                        break
            return out

        full = {1: te.get(1, ([], []))[0]}
        meta = {}  # te_id -> (parent, global branch ply, tail)
        for k in sorted(te):
            if k == 1:
                continue
            for (n, raw_m), pairs in sorted(bunki.items()):
                for i in range(0, len(pairs) - 1, 2):
                    if pairs[i] == k and n in full:
                        # bunki counts the parent's own moves; the parent's
                        # full line may carry a shared prefix before them
                        offset = len(full[n]) - len(te[n][0])
                        glob = offset + prefix_len(n, raw_m)
                        tail = tail_from(k, pairs[i + 1])
                        full[k] = full[n][:glob] + tail
                        meta[k] = (n, glob, tail)
                        break
                if k in full:
                    break
        main = (full[1], raw_comments(1))
        variations = []
        for k in sorted(te):
            if k == 1:
                continue
            if k in meta:
                n, plen, tail = meta[k]
                variations.append({"parent": n, "branch_after_ply": plen,
                                   "moves": tail, "full_line": full[k],
                                   "comments": raw_comments(k)})
            else:  # no bunki reference: fall back to engine search later
                variations.append({"parent": None, "branch_after_ply": None,
                                   "moves": te[k][0], "full_line": None,
                                   "comments": raw_comments(k)})
        return title, board, 'te', (main, variations)
    if not board:
        raise ValueError("no board data (no shoki[] arrays)")
    return title, board, 'sk', []  # position-only page (solution elsewhere)


def board_to_sfen(board):
    rows = []
    for y in range(12):
        row, run = '', 0
        for x in range(11, -1, -1):
            tok = board.get((x, y))
            if tok is None:
                run += 1
            else:
                if run:
                    row += str(run)
                    run = 0
                row += tok
        if run:
            row += str(run)
        rows.append(row)
    return '/'.join(rows)


class EngineDied(Exception):
    pass


class Engine:
    def __init__(self, binary):
        self.proc = subprocess.Popen(
            [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)

    def cmd(self, line):
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
            out = self.proc.stdout.readline()
        except (BrokenPipeError, ValueError):
            raise EngineDied(line)
        if not out:
            raise EngineDied(line)
        return out.rstrip("\n")

    def legal(self, sfen, moves=()):
        tail = " moves " + " ".join(moves) if moves else ""
        if self.cmd(f"position sfen {sfen}{tail}") != "ok":
            return None
        return self.cmd("legal").split()[1:]

    def mate(self, sfen, nodes):
        self.cmd(f"position sfen {sfen}")
        verdict = self.cmd(f"mate {nodes}")
        nodes_line = self.proc.stdout.readline().rstrip("\n")
        pv = []
        if verdict == "mate yes":
            pv = self.proc.stdout.readline().split()[1:]
        return verdict.split()[1], int(nodes_line.split()[1]), pv

    def close(self):
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        self.proc.wait()


def line_legal(eng, sfen, prefix, moves):
    """Every ply of `moves`, played after `prefix`, must be legal."""
    seq = list(prefix)
    for mv in moves:
        legal = eng.legal(sfen, tuple(seq))
        if not legal or mv not in legal:
            return False
        seq.append(mv)
    return True


def load_text(ref):
    """Load a page/kifu file (local path or URL); follow a kifu/*.txt
    script reference when the page itself holds no board data."""
    if ref.startswith(("http://", "https://")):
        with urllib.request.urlopen(ref) as r:
            raw = r.read()
        base = ref.rsplit("/", 1)[0] + "/"
    else:
        raw = open(ref, 'rb').read()
        base = ref.rsplit("/", 1)[0] + "/" if "/" in ref else "./"
    txt = decode_page(raw)
    if 'shoki[' not in txt:
        m = re.search(r'src="(?:\./)?(kifu/[^"]+\.txt)"', txt)
        if not m:
            raise ValueError(f"{ref}: no shoki[] board and no kifu reference")
        return load_text(base + m.group(1))
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page", help="problem page URL, local HTML file, or kifu txt")
    ap.add_argument("--answers", help="comma-separated extra answer pages/kifu "
                                    "files whose solution lines are appended "
                                    "as variations (chush2.js pages)")
    ap.add_argument("--id", help="record id (default: page file stem)")
    ap.add_argument("--side", choices=["b", "w"], default="b",
                    help="side to move; use w for retro-analysis problems "
                         "(曲詰) where gote is to move — the listed line is "
                         "then the illegal sente-first trap and is recorded "
                         "but not validated")
    ap.add_argument("--binary", help="engine binary for validation")
    ap.add_argument("--mate-nodes", type=int, default=1000000,
                    help="mate solver node cap; 0 disables the solver")
    ap.add_argument("--outdir", help="write <id>.json here instead of stdout")
    a = ap.parse_args()

    stem = a.page.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    txt = load_text(a.page)
    title, board, fmt, parsed = parse_page(txt)
    sfen = board_to_sfen(board) + f" {a.side} - 1"

    if fmt == 'sk':
        all_lines = list(parsed)
        for ref in (a.answers.split(',') if a.answers else []):
            atxt = load_text(ref.strip())
            atitle, aboard, afmt, alines = parse_page(atxt)
            if afmt != 'sk':
                raise ValueError(f"{ref}: answer page is not chush2.js format")
            if aboard != board:
                raise ValueError(f"{ref}: answer board differs from problem")
            all_lines.extend(alines)
        all_lines = [(mv, cm) for mv, cm in all_lines if mv]
        if not all_lines:
            main_line, main_cms = [], []
        else:
            main_line, main_cms = all_lines[0]
        variations = []
        for moves, cms in all_lines[1:]:
            # first divergence from the main line = branch point
            k = 0
            while k < len(main_line) and k < len(moves) \
                    and moves[k] == main_line[k]:
                k += 1
            variations.append({"parent": 1, "branch_after_ply": k,
                               "moves": moves[k:],
                               "full_line": main_line[:k] + moves[k:],
                               "comments": {str(i + 1 - k): c
                                            for i, c in enumerate(cms[k:], k)
                                            if c}})
        sol = {"moves": main_line,
               "comments": {str(i + 1): c for i, c in enumerate(main_cms) if c}}
    else:  # te format: variations resolved via the page's bunki data
        (main_line, sol_cms), variations = parsed
        sol = {"moves": main_line, "comments": sol_cms}

    rec = {
        "id": a.id or stem,
        "source": a.page,
        "title": title,
        "format": fmt,
        "sfen": sfen,
        "to_move": "sente" if a.side == "b" else "gote",
        "pieces": len(board),
        "solution": sol,
        "variations": variations,
    }
    if a.side == "w":
        rec["conditional"] = True
        rec["note"] = ("retro-analysis problem (曲詰): gote is to move; the "
                       "listed line is the illegal sente-first trap")

    if a.binary and a.side == "b":
        eng = Engine(a.binary)
        chk = {}
        try:
            chk["sfen_roundtrip"] = \
                eng.cmd(f"position sfen {sfen}") == "ok" and \
                eng.cmd("sfen") == "sfen " + sfen
            chk["solution_legal"] = line_legal(eng, sfen, (), main_line)
            for var in rec["variations"]:
                full = var.get("full_line")
                if full is not None:
                    var["legal"] = line_legal(eng, sfen, (), full)
                    continue
                mv = var["moves"]
                # no bunki data: the branch point is the main-line prefix
                # from which the whole tail replays legally
                var["legal"] = False
                for k in range(len(main_line), -1, -1):
                    if line_legal(eng, sfen, main_line[:k], mv):
                        var["branch_after_ply"] = k
                        var["full_line"] = main_line[:k] + mv
                        var["legal"] = True
                        break
            if a.mate_nodes:
                verdict, nodes, pv = eng.mate(sfen, a.mate_nodes)
                chk["mate"] = {"result": verdict, "nodes": nodes, "pv": pv}
        except EngineDied:
            chk["engine_died"] = True
        finally:
            eng.close()
        rec["engine_check"] = chk

    out = json.dumps(rec, ensure_ascii=False, indent=2) + "\n"
    if a.outdir:
        path = f"{a.outdir}/{rec['id']}.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"wrote {path}")
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
