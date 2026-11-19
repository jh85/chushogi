"""USI opening sampling: defaults, cutoff, seeds, colors and mate precedence."""
from pathlib import Path
import queue
import subprocess
import threading
import unittest

from az_blunder_test import SENTE, SAFE, mirror_sfen, mirror_move

INITIAL = ('lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/'
           '12/3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1')


class UsiSamplingTest(unittest.TestCase):
    def setUp(self):
        self.process = subprocess.Popen([str(Path('chushogi-az').resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True)
        self.output = queue.Queue()
        def drain():
            for line in self.process.stdout:
                self.output.put(line.strip())
        self.thread = threading.Thread(target=drain, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)
        self.send('usi')
        self.until('usiok')
        for name, value in [('Eval', 'random'), ('Batch', 8),
                            ('MateNodes', 0), ('ProbeDepth', 0)]:
            self.option(name, value)

    def send(self, text):
        self.process.stdin.write(text + '\n')
        self.process.stdin.flush()

    def until(self, prefix):
        lines = []
        while True:
            line = self.output.get(timeout=20)
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def option(self, name, value):
        self.send(f'setoption name {name} value {value}')

    def search(self, sfen=INITIAL, seed=None, new_game=True):
        if seed is not None:
            self.option('Seed', seed)
        if new_game:
            self.send('usinewgame')
        self.send('position sfen ' + sfen)
        self.send('go nodes 96')
        lines = self.until('bestmove ')
        return lines[-1].split()[1], lines

    def close(self):
        if self.process.poll() is None:
            self.send('quit')
            self.process.wait(timeout=20)
        self.process.stdin.close()
        self.process.stdout.close()
        self.thread.join(timeout=1)

    def test_advertised_defaults(self):
        self.send('usi')
        lines = self.until('usiok')
        self.assertIn('option name OpeningSamplingPlies type spin default 16 min 0 max 1000', lines)
        self.assertIn('option name OpeningSamplingTemp type string default 0.5', lines)
        self.assertIn('option name Seed type spin default 0 min 0 max 2147483647', lines)

    def test_opening_diversity_and_fixed_seed_reproduction(self):
        moves = {self.search(seed=seed)[0] for seed in range(1, 9)}
        self.assertGreater(len(moves), 1)
        repeated = [self.search(seed=17)[0] for _ in range(3)]
        self.assertEqual(len(set(repeated)), 1)

    def test_automatic_seed_changes_each_game_and_is_replayable(self):
        games = [self.search() for _ in range(3)]
        seeds = [int(next(s for s in lines if s.startswith('info string game seed ')).split()[-1])
                 for _, lines in games]
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertTrue(all(seed > 0 for seed in seeds))
        self.assertEqual(self.search(seed=seeds[0])[0], games[0][0])

    def test_cutoff_is_absolute_ply_and_zero_disables(self):
        self.option('OpeningSamplingPlies', 4)
        # A bare SFEN's move counter counts: do not restart sampling on analysis.
        _, before = self.search(INITIAL.rsplit(' ', 1)[0] + ' 4', seed=1)
        self.assertTrue(any('opening sampling: ply 3 of 4' in s for s in before))
        late = INITIAL.rsplit(' ', 1)[0] + ' 5'
        moves = []
        for seed in range(1, 4):
            best, lines = self.search(late, seed=seed)
            moves.append(best)
            self.assertFalse(any('opening sampling:' in s for s in lines))
        self.assertEqual(len(set(moves)), 1)
        self.option('OpeningSamplingPlies', 0)
        disabled = [self.search(seed=seed) for seed in range(1, 4)]
        self.assertEqual(len({best for best, _ in disabled}), 1)
        self.assertFalse(any('opening sampling:' in s for _, lines in disabled for s in lines))

    def test_white_sampling_converts_back_to_original_board(self):
        # Canonicalization reflects ranks and swaps colors, preserving files.
        board, _, lion, ply = INITIAL.split()
        white = '/'.join(board.split('/')[::-1]).swapcase() + f' w {lion} {ply}'
        ranks = str.maketrans('abcdefghijkl', 'lkjihgfedcba')
        for seed in range(1, 5):
            black_move, _ = self.search(seed=seed)
            white_move, _ = self.search(white, seed=seed)
            self.assertEqual(white_move, black_move.translate(ranks))

    def test_mate_override_has_priority_during_sampling(self):
        self.option('MateNodes', 5000)
        sfen = '12/6g5/7k4/12/12/12/12/12/12/12/12/7R3K b - 1'
        for board, expected in [(sfen, '5l5c'), (mirror_sfen(sfen), mirror_move('5l5c'))]:
            for seed in range(1, 4):
                self.assertEqual(self.search(board, seed=seed)[0], expected)

    def test_sampling_preserves_royal_safety_with_probes_on_and_off(self):
        self.option('OpeningSamplingPlies', 1000)
        for depth in [0, 3]:
            self.option('ProbeDepth', depth)
            for board, safe in [(SENTE, SAFE), (mirror_sfen(SENTE), {mirror_move(m) for m in SAFE})]:
                for seed in range(1, 4):
                    self.assertIn(self.search(board, seed=seed)[0], safe)

    def test_invalid_temperature_is_rejected(self):
        for invalid in ['0', '-1', 'nan', 'inf', 'garbage', '1junk']:
            self.option('OpeningSamplingTemp', invalid)
            _, lines = self.search(seed=1)
            self.assertTrue(any('OpeningSamplingTemp must be' in s for s in lines))
            self.assertTrue(any('temperature 0.5 ' in s for s in lines))


if __name__ == '__main__':
    unittest.main()
