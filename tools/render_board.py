#!/usr/bin/env python3
"""Render this repository's 12x12 chu-shogi SFEN as an English or Japanese-piece PNG.

Requires Pillow: python3 -m pip install Pillow
"""
import argparse
from pathlib import Path
import re
import sys
import subprocess


INITIAL_SFEN = (
    "lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/"
    "3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1"
)
PIECES = frozenset("pliamvcsftgebrhdoxknq")
PROMOTABLE = frozenset("pliamvcsftgbrhdoxe")


# Single-character display abbreviations; promoted roles that move identically
# share their glyph (e.g. a promoted kirin and an original lion are both 獅).
JP_PIECES = {
    "p": "歩", "l": "香", "i": "仲", "a": "反", "m": "横", "v": "竪",
    "c": "銅", "s": "銀", "f": "豹", "t": "虎", "g": "金", "e": "象",
    "b": "角", "r": "飛", "h": "馬", "d": "龍", "o": "麒", "x": "鳳",
    "k": "王", "n": "獅", "q": "奔",
    "+p": "と", "+l": "駒", "+i": "象", "+a": "鯨", "+m": "猪",
    "+v": "牛", "+c": "横", "+s": "竪", "+f": "角", "+t": "鹿",
    "+g": "飛", "+b": "馬", "+r": "龍", "+h": "鷹", "+d": "鷲",
    "+o": "獅", "+x": "奔", "+e": "太",
}


JP_FULL_PIECES = {
    "p": "歩兵", "l": "香車", "i": "仲人", "a": "反車", "m": "横行",
    "v": "竪行", "c": "銅将", "s": "銀将", "f": "猛豹", "t": "盲虎",
    "g": "金将", "e": "酔象", "b": "角行", "r": "飛車", "h": "龍馬",
    "d": "龍王", "o": "麒麟", "x": "鳳凰", "k": "王将", "n": "獅子",
    "q": "奔王", "+p": "と", "+l": "白駒", "+i": "酔象", "+a": "鯨鯢",
    "+m": "奔猪", "+v": "飛牛", "+c": "横行", "+s": "竪行",
    "+f": "角行", "+t": "飛鹿", "+g": "飛車", "+b": "龍馬",
    "+r": "龍王", "+h": "角鷹", "+d": "飛鷲", "+o": "獅子",
    "+x": "奔王", "+e": "太子",
}


def japanese_font_path():
    """Find a Japanese font without silently substituting an ASCII font."""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    try:
        result = subprocess.run(
            ["fc-list", ":lang=ja", "--format=%{file}\n"],
            capture_output=True, text=True, timeout=5, check=True)
        for candidate in result.stdout.splitlines():
            if Path(candidate).is_file():
                return candidate
    except (OSError, subprocess.SubprocessError):
        pass
    raise ValueError("jp mode needs a Japanese font; install Noto CJK fonts "
                     "or provide --font /path/to/Japanese-font.ttf")


def parse_sfen(sfen):
    """Return (rows in display order, turn, lion-capture square, move number).

    As in src/sfen.cpp, omitted trailing fields default to b, -, and 1.
    Require all twelve rows to contain exactly twelve squares.
    """
    fields = sfen.split()
    if not 1 <= len(fields) <= 4:
        raise ValueError("expected board [b|w [lion-capture-square|- [move-number]]]")
    board = fields[0]
    turn = fields[1] if len(fields) > 1 else "b"
    lioncap = fields[2] if len(fields) > 2 else "-"
    number = fields[3] if len(fields) > 3 else "1"
    if turn not in ("b", "w"):
        raise ValueError("side to move must be b or w")
    if lioncap != "-" and not re.fullmatch(r"(?:[1-9]|1[0-2])[a-l]", lioncap):
        raise ValueError("lion-capture field must be - or a square such as 12a")
    if not re.fullmatch(r"[0-9]+", number) or int(number) < 1:
        raise ValueError("move number must be a positive integer")
    ranks = board.split("/")
    if len(ranks) != 12:
        raise ValueError("board must have 12 ranks separated by /")
    rows = []
    for rank, encoded in enumerate(ranks):
        row = []
        i = 0
        while i < len(encoded):
            c = encoded[i]
            if c in "0123456789":
                end = i + 1
                while end < len(encoded) and encoded[end] in "0123456789":
                    end += 1
                run = int(encoded[i:end])
                if not 1 <= run <= 12:
                    raise ValueError("empty-square runs must be between 1 and 12")
                row.extend([""] * run)
                i = end
            else:
                promoted = c == "+"
                if promoted:
                    i += 1
                if i >= len(encoded) or encoded[i].lower() not in (
                        PROMOTABLE if promoted else PIECES):
                    raise ValueError(f"invalid piece in rank {chr(97 + rank)}")
                row.append(("+" if promoted else "") + encoded[i])
                i += 1
            if len(row) > 12:
                raise ValueError(f"rank {chr(97 + rank)} exceeds 12 squares")
        if len(row) != 12:
            raise ValueError(f"rank {chr(97 + rank)} must contain 12 squares")
        rows.append(row)
    return rows, turn, lioncap, int(number)


def render_board(sfen, cell_size=48, font_path=None, mode="en"):
    """Return an RGB image; Japanese modes rotate White 180 degrees.

    en preserves ASCII; jp uses single glyphs; jp-full stacks full names.
    For Japanese modes, font_path
    must cover the Japanese glyphs if provided; otherwise auto-detect a font.
    """
    from PIL import Image, ImageDraw, ImageFont

    if mode not in ("en", "jp", "jp-full"):
        raise ValueError("mode must be en, jp, or jp-full")
    if cell_size < 24:
        raise ValueError("cell size must be at least 24 pixels")
    rows, turn, _, _ = parse_sfen(sfen)

    def font(size):
        if font_path:
            return ImageFont.truetype(str(font_path), size)
        for candidate in ("DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
        # Pillow >= 10.1 supplies a scalable bundled ASCII font.
        return ImageFont.load_default(size=size)

    piece_font = (ImageFont.truetype(
        str(font_path or japanese_font_path()),
        round(cell_size * (0.39 if mode == "jp-full" else 0.65)))
        if mode != "en" else font(round(cell_size * 0.65)))
    label_font = font(round(cell_size * 0.44))
    margin = cell_size
    edge = margin + 12 * cell_size
    img = Image.new("RGB", (14 * cell_size, 14 * cell_size), "white")
    draw = ImageDraw.Draw(img)
    width = max(1, cell_size // 24)
    for i in range(13):
        xy = margin + i * cell_size
        draw.line((xy, margin, xy, edge), fill="black", width=width)
        draw.line((margin, xy, edge, xy), fill="black", width=width)
    for col in range(12):
        draw.text((margin + (col + 0.5) * cell_size, margin / 2),
                  str(12 - col), font=label_font, fill="black", anchor="mm")
    for rank, row in enumerate(rows):
        y = margin + (rank + 0.5) * cell_size
        draw.text((edge + margin / 2, y), chr(97 + rank),
                  font=label_font, fill="black", anchor="mm")
        for col, piece in enumerate(row):
            if piece:
                if mode == "en":
                    draw.text((margin + (col + 0.5) * cell_size, y), piece,
                              font=piece_font, fill="black", anchor="mm")
                else:
                    # Center the actual ink, then rotate the entire transparent
                    # tile so both colors have exactly the same placement.
                    glyph = (JP_FULL_PIECES if mode == "jp-full" else JP_PIECES)[piece.lower()]
                    tile = Image.new("RGBA", (cell_size, cell_size))
                    ink = ImageDraw.Draw(tile)
                    if mode == "jp-full":
                        # Stack the name vertically, keeping the whole stack
                        # centered. A one-character name remains centered too.
                        step = cell_size * 0.43
                        for index, char in enumerate(glyph):
                            left, top, right, bottom = ink.textbbox((0, 0), char, font=piece_font)
                            center_y = cell_size / 2 + (index - (len(glyph) - 1) / 2) * step
                            ink.text(((cell_size - right - left) / 2,
                                      center_y - (bottom + top) / 2), char,
                                     font=piece_font, fill="black")
                    else:
                        left, top, right, bottom = ink.textbbox((0, 0), glyph, font=piece_font)
                        ink.text(((cell_size - right - left) / 2,
                                  (cell_size - bottom - top) / 2), glyph,
                                 font=piece_font, fill="black")
                    if piece[-1].islower():
                        tile = tile.transpose(Image.Transpose.ROTATE_180)
                    img.paste(tile, (margin + col * cell_size,
                                     margin + rank * cell_size), tile)
    caption = ("Black" if turn == "b" else "White") + " to move"
    draw.text((img.width / 2, edge + margin / 2), caption,
              font=label_font, fill="black", anchor="mm")
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sfen", nargs="?", help="quoted SFEN; omit for the initial position")
    parser.add_argument("-o", "--output", type=Path, default=Path("board.png"))
    parser.add_argument("--cell-size", type=int, default=48, help="square size in pixels (default: 48)")
    parser.add_argument("--font", type=Path, help="optional TrueType/OpenType font")
    parser.add_argument("--mode", choices=("en", "jp", "jp-full"), default="en",
                        help="piece symbols: en (default ASCII), jp (single character), jp-full (vertical names)")
    args = parser.parse_args()
    try:
        img = render_board(args.sfen if args.sfen is not None else INITIAL_SFEN,
                           args.cell_size, args.font, mode=args.mode)
        img.save(args.output, format="PNG")
    except ImportError:
        parser.exit(2, "Pillow is required: python3 -m pip install 'Pillow>=10.1'\n")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
