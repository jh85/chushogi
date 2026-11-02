import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'az'))
from selfplay_pool import game_jobs,run_dynamic

class PoolTest(unittest.TestCase):
    def test_seeds_and_exactly_once_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);engine=root/'engine'
            engine.write_text('''#!/usr/bin/env python3
import sys,json,pathlib,time
args=sys.stdin.readline().split()[1:];a=dict(zip(args[::2],args[1::2]))
seed=int(a['--seed'])+1
time.sleep(.01*(seed%3))
p=pathlib.Path(a['--out']);p.mkdir(parents=True)
(p/'game_test.json').write_text(json.dumps(dict(seed=seed)))
print('game 0: 12 plies, fourfold, result 0',flush=True)
''');engine.chmod(0o755)
            plies,count=run_dynamic(str(engine),[123,456],4,2,17,9,root/'games','--sims 16')
            self.assertEqual((plies,count),(96,8))
            for w,g,seed in game_jobs(4,2,17,9):
                rec=json.loads((root/'games'/f'worker_{w:04d}'/f'game_{g:04d}'/'game_test.json').read_text())
                self.assertEqual(rec['seed'],17+9*100003+w*101+g+1)
            self.assertEqual(len(list((root/'games').rglob('game_*.json'))),8)

    def test_failed_worker_aborts_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad';p.write_text('#!/bin/sh\nexit 3\n');p.chmod(0o755)
            with self.assertRaisesRegex(RuntimeError,'failed'):
                run_dynamic(str(p),[1],2,2,1,0,Path(tmp)/'out','')

if __name__=='__main__':unittest.main()
