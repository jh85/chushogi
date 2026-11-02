"""Ensure copied anchors do not create self-matches or duplicate opponents."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'az'))
from checkpoint_tournament import unique_checkpoints, search_command


class TournamentTest(unittest.TestCase):
    def test_search_command_matches_selfplay_budgets(self):
        self.assertEqual(search_command(1024),
            'go az --sims 1024 --mate-nodes 5000 --probe-depth 3 --probe-nodes 2000')
        self.assertEqual(search_command(256,0,0,2000),
            'go az --sims 256 --mate-nodes 0 --probe-depth 0 --probe-nodes 2000')
        self.assertIn('--probe-nodes 123',search_command(1024,5000,3,123))

    def test_copied_anchor_and_same_path_are_one_opponent(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ('anchor.pt', 'iter_0000.pt', 'iter_0009.pt')]
            for path, content in zip(paths, (b'early', b'early', b'later')):
                path.write_bytes(content)
            self.assertEqual(unique_checkpoints(paths + [paths[0]]), [paths[0], paths[2]])

    def test_identical_snapshots_skip_engines(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'checkpoint.pt'
            checkpoint.write_bytes(b'identical snapshot; intentionally not a torch file')
            output = Path(tmp) / 'results'
            subprocess.run([sys.executable, str(ROOT / 'az/checkpoint_tournament.py'),
                            '--checkpoints', str(checkpoint), str(checkpoint),
                            '--az', '/nonexistent-engine', '--out', str(output)],
                           check=True, capture_output=True, text=True, timeout=10)
            self.assertEqual(json.loads((output / 'summary.json').read_text()), [])


if __name__ == '__main__':
    unittest.main()
