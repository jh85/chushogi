import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'az'))
from tune_inference_servers import choose_winner,resume_command,positions_from_iteration

class TuningTest(unittest.TestCase):
    def test_median_and_tie_break(self):
        self.assertEqual(choose_winner({2:[2,100,2],3:[3,3,3],4:[4,4,4]}),2)
        self.assertEqual(choose_winner({4:[2,2],3:[2,2],2:[2,2]}),2)

    def test_resume_keeps_learning_settings_and_target(self):
        config=dict(python=sys.executable,run_dir='/data/run',az='/bin/az',init_from='',resume=True,
                    iterations=59,workers=44,games=2,sims=256,batch=16,root_policy_temp=1.25,
                    temp_plies=-1,lr=.001,lr_decay_steps=200000,dynamic_games=True,eval_servers=1)
        cmd=resume_command(config,81,139,3)
        self.assertEqual(cmd[cmd.index('--iterations')+1],'58')
        self.assertEqual(cmd[cmd.index('--eval-servers')+1],'3')
        self.assertEqual(cmd[cmd.index('--temp-plies')+1],'-1')
        self.assertEqual(cmd[cmd.index('--root-policy-temp')+1],'1.25')
        self.assertIn('--dynamic-games',cmd)
        self.assertNotIn('--init-from',cmd)
        self.assertEqual(config['eval_servers'],1)
        with self.assertRaises(ValueError):resume_command(config,139,139,2)

    def test_histories_from_completed_iteration_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for it in [80,81]:
                folder=root/'games'/f'iter_{it:06d}';folder.mkdir(parents=True)
                for i in range(4):
                    (folder/f'game_{i}.json').write_text(json.dumps(dict(initial='initial',moves=['a','b','c','d'])))
            positions=positions_from_iteration(root,80,3)
            self.assertEqual(len(positions),3)
            self.assertEqual(len({p['id'] for p in positions}),3)
            self.assertTrue(all('iter_000080' in p['source'] and 0<len(p['moves'])<4 for p in positions))
            self.assertEqual(len(positions_from_iteration(root,80,1)),1)

if __name__=='__main__':unittest.main()
