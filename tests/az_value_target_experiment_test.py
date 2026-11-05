"""Value-target branches must differ only in target and isolated output paths."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'az'))
from buffer import Buffer
from value_target_experiment import prepare_targets

class ValueTargetTest(unittest.TestCase):
    def test_matched_replay_optimizer_and_training_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'common';(source/'ckpt').mkdir(parents=True)
            (source/'ckpt/iter_0182.pt').write_bytes(b'weights-and-optimizer-at-36600')
            (source/'loop_config.json').write_text(json.dumps(dict(
                python=sys.executable,seed=1,lr=.001,lr_decay_steps=200000,
                train_steps=200,train_batch=256,max_positions=1000000,
                value_target='z',mate_nodes=5000,probe_depth=3,probe_nodes=2000)))
            records=source/'games/iter_000182';records.mkdir(parents=True)
            (records/'game_0.json').write_text('{"samples":[]}')
            binary=root/'engine';binary.write_bytes(b'fixed-a0gb-engine')
            out=root/'experiment';m=prepare_targets(source,182,out,binary)
            self.assertEqual(set(m['run_order']),{'z','softz','a0gb'})
            self.assertEqual(m['pilot_iterations'],20)
            options=[]
            for branch in m['branches']:
                cmd=branch['command'][3:];o={};i=0
                bools={'--resume','--dynamic-games','--eval-tcp-nodelay'}
                while i<len(cmd):
                    flag=cmd[i]
                    if flag in bools:o[flag]=True;i+=1
                    else:o[flag]=cmd[i+1];i+=2
                self.assertEqual(o['--value-target'],branch['name'])
                self.assertEqual((o['--sims'],o['--workers'],o['--games'],o['--eval-servers']),('1024','44','2','4'))
                self.assertEqual(o['--train-steps'],'200')
                self.assertEqual(o['--lr-decay-steps'],'200000')
                run=Path(branch['run_dir'])
                self.assertEqual((run/'ckpt/current.pt').read_bytes(),b'weights-and-optimizer-at-36600')
                rec=m['replay'][0]
                self.assertEqual((run/'games'/rec['path']).stat().st_mtime_ns,rec['mtime_ns'])
                options.append(o)
            for other in options[1:]:
                self.assertEqual({k for k in options[0] if options[0][k]!=other[k]},
                                 {'--run-dir','--stop-file','--value-target'})
            (out/'z/games/iter_000182/game_0.json').write_text('changed')
            self.assertEqual((out/'softz/games/iter_000182/game_0.json').read_text(),'{"samples":[]}')
            self.assertEqual((records/'game_0.json').read_text(),'{"samples":[]}')

    def test_target_distributions_and_draw_encoding(self):
        buf=Buffer.__new__(Buffer)
        buf.z=np.array([1.,.5,0.],dtype=np.float32)
        buf.rootq=np.array([.8,.5,.1],dtype=np.float32)
        buf.a0gb=np.array([.7,.3,.2],dtype=np.float32)
        for mode,expected in [('z',[[1,0,0],[0,1,0],[0,0,1]]),
                              ('softz',[[.8,0,.2],[.5,0,.5],[.1,0,.9]]),
                              ('a0gb',[[.7,0,.3],[.3,0,.7],[.2,0,.8]])]:
            got=buf.value_targets(mode,.5)
            np.testing.assert_allclose(got,expected,atol=1e-7)
            np.testing.assert_allclose(got.sum(axis=1),1)
            self.assertTrue(np.isfinite(got).all())

if __name__=='__main__':unittest.main()
