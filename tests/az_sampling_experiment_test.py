import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'az'))
from sampling_experiment import prepare_sampling

class SamplingExperimentTest(unittest.TestCase):
    def test_matched_branches_and_greedy_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';(source/'ckpt').mkdir(parents=True)
            (source/'ckpt/iter_0001.pt').write_bytes(b'model-and-optimizer')
            config=dict(seed=1,workers=44,games=2,eval_servers=1,sims=256,batch=16,
                        train_steps=200,lr=.001,lr_decay_steps=200000,root_policy_temp=1.25,
                        dynamic_games=True,eval_tcp_nodelay=True,eval_max_batch=128,python=sys.executable)
            (source/'loop_config.json').write_text(json.dumps(config))
            for it in range(3):
                folder=source/'games'/f'iter_{it:06d}';folder.mkdir(parents=True)
                (folder/'game_test.json').write_text(json.dumps(dict(iteration=it)))
            binary=root/'engine';binary.write_bytes(b'engine')
            out=root/'experiment';manifest=prepare_sampling(source,1,out,binary)
            self.assertEqual(len(manifest['replay']),2)
            def options(cmd):
                result={};i=2
                while i<len(cmd):
                    key=cmd[i];i+=1
                    if key in ('--resume','--dynamic-games','--eval-tcp-nodelay'):result[key]=True
                    else:result[key]=cmd[i];i+=1
                return result
            a,b=[options(branch['command']) for branch in manifest['branches']]
            self.assertEqual({k for k in a if a[k]!=b[k]}, {'--temp-plies','--run-dir','--stop-file'})
            self.assertEqual((a['--temp-plies'],b['--temp-plies']),('20','-1'))
            self.assertEqual(a['--root-policy-temp'],'1.25')
            self.assertEqual((a['--workers'],a['--games']),('22','4'))
            self.assertTrue(a['--dynamic-games'])
            self.assertEqual(a['--eval-max-batch'],'128')
            for branch in ('opening20','all_moves'):
                self.assertEqual((out/branch/'ckpt/current.pt').read_bytes(),b'model-and-optimizer')
                self.assertFalse((out/branch/'games/iter_000002').exists())
                rec=manifest['replay'][0]
                self.assertEqual((out/branch/'games'/rec['path']).stat().st_mtime_ns,rec['mtime_ns'])
            evaluation=json.loads((out/'evaluation_command.json').read_text())
            self.assertIn(str(out/'opening20/ckpt/iter_0021.pt'),evaluation)
            self.assertIn(str(out/'all_moves/ckpt/iter_0021.pt'),evaluation)
            self.assertNotIn('--temp-plies',evaluation)
            self.assertNotIn('--root-policy-temp',evaluation)
            self.assertEqual(evaluation[evaluation.index('--pairs')+1],'50')
            self.assertEqual(evaluation[evaluation.index('--sims')+1],'1024')
            (out/'opening20/games/iter_000000/game_test.json').write_text('changed')
            self.assertNotEqual((out/'all_moves/games/iter_000000/game_test.json').read_text(),'changed')
            self.assertNotEqual((source/'games/iter_000000/game_test.json').read_text(),'changed')

if __name__=='__main__':unittest.main()
