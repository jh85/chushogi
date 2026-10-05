"""AlphaZero input encoding for chu shogi — Python mirror of src/az/encode.cpp.

Positions are canonicalized to the side-to-move's perspective (vertical
mirror + color swap when gote is to move), then encoded as 82 planes of
12x12:

  planes  0..38 : our pieces, one plane per role (order = C++ Role enum)
  planes 39..77 : their pieces
  plane  78     : one-hot on lastLionCapture (lion-immunity game state)
  plane  79     : broadcast 1 if the position occurred once before
  plane  80     : broadcast 1 if it occurred twice or more before
  plane  81     : broadcast ply / 1000 (1000-ply draw-cap progress)

Policy space (50,688 logits): from x to (144x144 = 20,736), a second
from x to matrix for promotions, and a from x midDir(8) x toDir(8) block
(9,216) for two-step lion-power moves (lion / horned falcon / soaring
eagle, incl. igui). Moves must be in the canonical (flipped) frame.
"""

import numpy as np

FILES = RANKS = 12
BOARD = FILES * RANKS
NUM_PLANES = 82

POLICY_MOVE = BOARD * BOARD          # 20,736
POLICY_DOUBLE = BOARD * 8 * 8        # 9,216
POLICY_SIZE = 2 * POLICY_MOVE + POLICY_DOUBLE

# C++ Role enum order.
ROLES = [
    'Pawn', 'Lance', 'GoBetween', 'Chariot', 'SideMover', 'VerticalMover',
    'Copper', 'Silver', 'Leopard', 'Tiger', 'Gold', 'Elephant',
    'Bishop', 'Rook', 'Horse', 'Dragon', 'Kirin', 'Phoenix',
    'King', 'Lion', 'Queen',
    'Tokin', 'WhiteHorse', 'ElephantP', 'Whale', 'Boar', 'Ox',
    'SideMoverP', 'VerticalMoverP', 'BishopP', 'Stag', 'RookP',
    'HorseP', 'DragonP', 'Falcon', 'Eagle', 'LionP', 'QueenP', 'Prince',
]
ROLE_INDEX = {r: i for i, r in enumerate(ROLES)}

# SFEN letter -> role (same letters as src/sfen.cpp).
_SFEN_PLAIN = dict(zip('pliamvcsftgebrhdoxknq',
                       ['Pawn', 'Lance', 'GoBetween', 'Chariot', 'SideMover',
                        'VerticalMover', 'Copper', 'Silver', 'Leopard',
                        'Tiger', 'Gold', 'Elephant', 'Bishop', 'Rook',
                        'Horse', 'Dragon', 'Kirin', 'Phoenix', 'King',
                        'Lion', 'Queen']))
_SFEN_PROMOTED = dict(zip('pliamvcsftgbrhdoxe',
                          ['Tokin', 'WhiteHorse', 'ElephantP', 'Whale', 'Boar',
                           'Ox', 'SideMoverP', 'VerticalMoverP', 'BishopP',
                           'Stag', 'RookP', 'HorseP', 'DragonP', 'Falcon',
                           'Eagle', 'LionP', 'QueenP', 'Prince']))

_DIRS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
_DIR_INDEX = {d: i for i, d in enumerate(_DIRS)}


def parse_square(s):
    """'6c' -> square index (x = file - 1, y = rank - 'a')."""
    i = 0
    while s[i].isdigit():
        i += 1
    return (int(s[:i]) - 1) + (ord(s[i]) - ord('a')) * FILES


def square_name(sq):
    return f"{sq % FILES + 1}{chr(ord('a') + sq // FILES)}"


def flip_square(sq):
    return (RANKS - 1 - sq // FILES) * FILES + sq % FILES


def flip_usi_move(token):
    out = []
    i = 0
    while i < len(token):
        if token[i] == '+':
            out.append('+')
            i += 1
            continue
        j = i
        while j < len(token) and token[j].isdigit():
            j += 1
        file_, rank = token[i:j], token[j]
        out.append(file_ + chr(ord('a') + ord('l') - ord(rank)))
        i = j + 1
    return ''.join(out)


class Position:
    """Board state parsed from SFEN (no move generation here)."""

    def __init__(self, sfen):
        parts = sfen.split()
        board, self.side, lion, num = parts[0], parts[1], parts[2], parts[3]
        self.board = [None] * BOARD  # sq -> (role_index, color 0=sente 1=gote)
        x, y = FILES - 1, 0
        i = 0
        while i < len(board):
            c = board[i]
            if c == '/':
                y += 1
                x = FILES - 1
            elif c.isdigit():
                j = i
                while j < len(board) and board[j].isdigit():
                    j += 1
                x -= int(board[i:j])
                i = j - 1
            else:
                promoted = c == '+'
                if promoted:
                    i += 1
                    c = board[i]
                table = _SFEN_PROMOTED if promoted else _SFEN_PLAIN
                role = ROLE_INDEX[table[c.lower()]]
                color = 0 if c.isupper() else 1
                self.board[y * FILES + x] = (role, color)
                x -= 1
            i += 1
        self.side_to_move = 0 if self.side == 'b' else 1
        self.last_lion_capture = None if lion == '-' else parse_square(lion)
        self.move_number = int(num)

    def flipped(self):
        out = Position.__new__(Position)
        out.board = [None] * BOARD
        for s, pc in enumerate(self.board):
            if pc:
                out.board[flip_square(s)] = (pc[0], pc[1] ^ 1)
        out.side = 'w' if self.side == 'b' else 'b'
        out.side_to_move = self.side_to_move ^ 1
        out.last_lion_capture = (None if self.last_lion_capture is None
                                 else flip_square(self.last_lion_capture))
        out.move_number = self.move_number
        return out


def move_to_index(usi):
    """USI move (canonical frame) -> policy index."""
    promote = usi.endswith('+')
    if promote:
        usi = usi[:-1]
    # squares: file digits + rank letter, 2 or 3 of them
    squares = []
    i = 0
    while i < len(usi):
        j = i
        while usi[j].isdigit():
            j += 1
        squares.append(parse_square(usi[i:j + 1]))
        i = j + 1
    if len(squares) == 2:
        idx = squares[0] * BOARD + squares[1]
        return POLICY_MOVE + idx if promote else idx
    f, mid, to = squares
    fx, fy = f % FILES, f // FILES
    mx, my = mid % FILES, mid // FILES
    tx, ty = to % FILES, to // FILES
    md = _DIR_INDEX[(mx - fx, my - fy)]
    td = _DIR_INDEX[(tx - mx, ty - my)]
    return 2 * POLICY_MOVE + f * 64 + md * 8 + td


def index_to_move(idx):
    """Policy index -> USI move (canonical frame)."""
    if idx < 2 * POLICY_MOVE:
        promote = idx >= POLICY_MOVE
        i = idx % POLICY_MOVE
        return square_name(i // BOARD) + square_name(i % BOARD) + \
            ('+' if promote else '')
    i = idx - 2 * POLICY_MOVE
    f, md, td = i // 64, (i // 8) % 8, i % 8
    fx, fy = f % FILES, f // FILES
    mx, my = fx + _DIRS[md][0], fy + _DIRS[md][1]
    tx, ty = mx + _DIRS[td][0], my + _DIRS[td][1]
    return square_name(f) + square_name(my * FILES + mx) + \
        square_name(ty * FILES + tx)


def encode_position(pos, rep_count=0, ply=None):
    """Position -> float32 array of shape (NUM_PLANES, 12, 12)."""
    if ply is None:  # moveNumber counts plies + 1
        ply = pos.move_number - 1
    if pos.side_to_move == 1:
        pos = pos.flipped()
    out = np.zeros((NUM_PLANES, BOARD), dtype=np.float32)
    for s, pc in enumerate(pos.board):
        if pc:
            role, color = pc
            plane = role if color == pos.side_to_move else 39 + role
            out[plane, s] = 1.0
    if pos.last_lion_capture is not None:
        out[78, pos.last_lion_capture] = 1.0
    if rep_count >= 1:
        out[79, :] = 1.0
    if rep_count >= 2:
        out[80, :] = 1.0
    out[81, :] = min(ply / 1000.0, 1.0)
    return out.reshape(NUM_PLANES, RANKS, FILES)
