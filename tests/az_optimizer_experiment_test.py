import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'az'))
from optimizer_experiment import choose_scale,prepare_branches

class ExperimentTest(unittest.TestCase):
    def test_selection_and_stability_gate(self):
        baseline=dict(optimizer='adamw',status='complete',scale=1.,val_loss=5.)
        rows=[baseline]+[dict(optimizer='muon_adamw',status='complete',scale=s,val_loss=v)
                         for s,v in [(.3,4.98),(1.,5.),(3.,6.)]]
        self.assertEqual(choose_scale(rows),1.)
        rows[1]['val_loss']=4.
        self.assertEqual(choose_scale(rows),.3)
        for row in rows[1:]:row['val_loss']=7.
        with self.assertRaisesRegex(RuntimeError,'gate failed'):choose_scale(rows)
        for row in rows[1:]:row['status']='failed'
        with self.assertRaisesRegex(RuntimeError,'gate failed'):choose_scale(rows)
    def test_optimizer_is_only_training_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);source=p/'source';(source/'ckpt').mkdir(parents=True)
            (source/'ckpt/iter_0202.pt').write_bytes(b'weights-and-adamw')
            (source/'loop_config.json').write_text(json.dumps(dict(python=sys.executable,
                lr=.001,train_steps=200,train_batch=256,lr_decay_steps=200000,seed=1,
                value_target='a0gb',mate_nodes=5000,probe_depth=3,probe_nodes=2000)))
            d=source/'games/iter_000202';d.mkdir(parents=True)
            (d/'game_0.json').write_text('{}')
            binary=p/'az';binary.write_bytes(b'fixed-engine')
            m=prepare_branches(source,202,p/'experiment',binary,20,.3)
            a,b=m['branches'];self.assertEqual(set(m['run_order']),{'adamw','muon_adamw'})
            self.assertEqual(m['pilot_iterations'],20)
            for branch in (a,b):
                self.assertEqual((Path(branch['run_dir'])/'ckpt/current.pt').read_bytes(),b'weights-and-adamw')
                self.assertEqual(branch['settings']['value_target'],'a0gb')
                self.assertEqual(branch['settings']['muon_warmup_steps'],50)
            aa=a['settings'];bb=b['settings']
            self.assertEqual({k for k in aa if aa[k]!=bb[k]},{'optimizer','muon_lr_scale'})

if __name__=='__main__':unittest.main()
