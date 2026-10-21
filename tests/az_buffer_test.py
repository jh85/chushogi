"""Replay-window regression: PID filenames must not determine data recency."""
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'az'))
from buffer import Buffer

class ReplayWindowTest(unittest.TestCase):
    def test_newest_positions_across_pid_wrap_and_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = [Path(tmp) / 'a', Path(tmp) / 'b']
            for d in dirs:
                d.mkdir()
            for name, stamp, d, values in [('game_999_0000.json', 10, dirs[0], [0, 0]),
                                          ('game_100_0000.json', 30, dirs[1], [1, 1]),
                                          ('game_800_0000.json', 20, dirs[0], [.5, .5])]:
                samples = [dict(ply=i+4, planes=base64.b64encode(bytes(1458)).decode(),
                                progress=0, policy=[[0, 1]], z=z, rootq=z, a0gb=z)
                           for i,z in enumerate(values)]
                p = d/name
                p.write_text(json.dumps(dict(samples=samples)))
                os.utime(p, (stamp, stamp))
            buf = Buffer(dirs, max_positions=3)
            self.assertEqual(buf.z.tolist(), [.5, 1, 1])
            self.assertEqual(len(buf), 3)
            self.assertEqual(buf.value_targets('z', .5).tolist(),
                             [[0, 1, 0], [1, 0, 0], [1, 0, 0]])

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            Buffer([], max_positions=0)

if __name__ == '__main__':
    unittest.main()
