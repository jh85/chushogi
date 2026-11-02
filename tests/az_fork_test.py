#!/usr/bin/env python3
"""Forks must preserve checkpoint/replay state while isolating future writes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'az'))
from fork_training import prepare


class ForkTest(unittest.TestCase):
    def test_matched_isolated_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / 'source'
            (source / 'ckpt').mkdir(parents=True)
            (source / 'ckpt/iter_0001.pt').write_bytes(b'checkpoint-and-optimizer')
            (source / 'loop_config.json').write_text(json.dumps(dict(
                seed=17, lr=0.001, train_steps=200, python=sys.executable,
                workers=44, games=2, eval_servers=4, run_dir=str(source), resume=False)))
            for it in range(3):
                d = source / 'games' / f'iter_{it:06d}'
                d.mkdir(parents=True)
                (d / 'game_1.json').write_text(json.dumps(dict(iteration=it)))
            binary = base / 'engine'
            binary.write_bytes(b'engine')
            out = base / 'fork'
            manifest = prepare(source, 1, out, binary, 20)
            self.assertEqual(len(manifest['replay']), 2)
            for tag in ('t1', 't1p25'):
                branch = out / tag
                self.assertFalse((branch / 'games/iter_000002').exists())
                self.assertEqual((branch / 'ckpt/current.pt').read_bytes(), b'checkpoint-and-optimizer')
                self.assertTrue((branch / 'ckpt/iter_0001.pt').exists())
                self.assertFalse((branch / 'ckpt/iter_0000.pt').exists())
                rec = manifest['replay'][0]
                self.assertEqual((branch / 'games' / rec['path']).stat().st_mtime_ns, rec['mtime_ns'])
            a, b = manifest['branches']
            def options(branch):
                cmd = branch['command'][3:]  # python, script, --resume
                return dict(zip(cmd[::2], cmd[1::2]))
            oa, ob = options(a), options(b)
            self.assertEqual({k for k in oa if oa[k] != ob[k]},
                             {'--run-dir', '--root-policy-temp', '--stop-file'})
            self.assertEqual(oa['--workers'], '22')
            self.assertEqual(oa['--games'], '4')
            self.assertEqual(oa['--seed'], '17')
            self.assertEqual(oa['--train-lock'], ob['--train-lock'])
            (out / 't1/games/iter_000000/game_1.json').write_text('changed')
            self.assertNotEqual((out / 't1p25/games/iter_000000/game_1.json').read_text(), 'changed')
            self.assertNotEqual((source / 'games/iter_000000/game_1.json').read_text(), 'changed')
            with self.assertRaises(FileExistsError):
                prepare(source, 1, out, binary)


if __name__ == '__main__':
    unittest.main()
