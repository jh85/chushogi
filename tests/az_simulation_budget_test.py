import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import time
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'az'))
import simulation_budget_experiment as experiment
from simulation_budget_experiment import prepare_experiment,select_checkpoint

class SimulationBudgetTest(unittest.TestCase):
    def test_equal_conditions_except_search_and_independent_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';(source/'ckpt').mkdir(parents=True)
            (source/'ckpt/iter_0139.pt').write_bytes(b'model-with-optimizer')
            (source/'loop_config.json').write_text(json.dumps(dict(
                python=sys.executable,seed=1,sims=256,train_steps=200,train_batch=256,
                max_positions=1000000,lr=.001,lr_decay_steps=200000,mate_nodes=5000,
                probe_depth=3,probe_nodes=2000,root_policy_temp=1.25,temp_plies=-1)))
            folder=source/'games/iter_000139';folder.mkdir(parents=True)
            (folder/'game_test.json').write_text('{}')
            engine=root/'engine';engine.write_bytes(b'engine')
            out=root/'experiment'
            manifest=prepare_experiment(source,139,out,engine,hours=12)
            self.assertEqual(manifest['hours_per_branch'],12)
            self.assertEqual(len(manifest['branches']),3)
            self.assertEqual(set(manifest['run_order']),{'sims_256','sims_1024','sims_4096'})
            self.assertFalse((out/'evaluation_command.json').exists())
            baseline=manifest['branches'][0]['options']
            for n,branch in zip([256,1024,4096],manifest['branches']):
                opts=branch['options']
                self.assertEqual(opts['sims'],n)
                self.assertEqual({k for k in opts if opts[k]!=baseline[k]},
                    set() if n==256 else {'sims','run_dir','stop_file'})
                self.assertEqual(opts['strength_every'],0)
                self.assertEqual((opts['workers'],opts['games'],opts['eval_servers']),(44,2,4))
                self.assertEqual(opts['temp_plies'],-1)
                self.assertEqual((Path(branch['run_dir'])/'ckpt/current.pt').read_bytes(),b'model-with-optimizer')
                rec=manifest['replay'][0]
                self.assertEqual((Path(branch['run_dir'])/'games'/rec['path']).stat().st_mtime_ns,rec['mtime_ns'])
            (out/'sims_256/games/iter_000139/game_test.json').write_text('changed')
            self.assertEqual((out/'sims_4096/games/iter_000139/game_test.json').read_text(),'{}')
            self.assertEqual((source/'games/iter_000139/game_test.json').read_text(),'{}')

    def test_sequential_deadlines_and_common_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=root/'fake_loop.py'
            script.write_text("""import sys,time
from pathlib import Path
run=Path(sys.argv[1]);iteration=139
while not (run/'STOP').exists():
 time.sleep(.03);iteration+=1
 (run/'ckpt'/f'iter_{iteration:04d}.pt').write_bytes(str(iteration).encode())
(run/'loop.log').write_text('[loop] finished\\n')
""")
            branches=[]
            for n in [256,1024,4096]:
                run=root/f'sims_{n}';(run/'ckpt').mkdir(parents=True)
                (run/'ckpt/iter_0139.pt').write_bytes(b'initial')
                opts=dict(python=sys.executable,stop_file=str(run/'STOP'),train_steps=200,
                          mate_nodes=5000,probe_depth=3,probe_nodes=2000)
                branches.append(dict(name=run.name,run_dir=str(run),command=[sys.executable,str(script),str(run)],options=opts))
            (root/'manifest.json').write_text(json.dumps(dict(branches=branches,source_iteration=139,
                run_order=[b['name'] for b in branches],hours_per_branch=.00004)))
            real_sleep=time.sleep
            with patch.object(experiment,'running_workloads',return_value=[]), \
                 patch.object(experiment.signal,'signal'), \
                 patch.object(experiment.time,'sleep',side_effect=lambda n:real_sleep(min(n,.02))), \
                 patch.object(experiment.subprocess,'run') as evaluation:
                evaluation.return_value.returncode=0
                experiment.run_experiment(root)
            status=json.loads((root/'experiment_status.json').read_text())
            self.assertEqual(status['status'],'complete')
            self.assertEqual(len(status['branches']),3)
            for branch in status['branches'].values():
                self.assertGreater(branch['completed_iterations_within_budget'],0)
                self.assertLessEqual(Path(branch['selected_checkpoint']).stat().st_mtime,branch['deadline'])
            command=json.loads((root/'evaluation_command.json').read_text())
            self.assertEqual(command[command.index('--sims')+1],'1024')
            self.assertEqual(evaluation.call_count,1)

    def test_checkpoints_after_deadline_are_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'ckpt').mkdir()
            for iteration,mtime in [(139,10),(140,99),(141,101)]:
                p=root/'ckpt'/f'iter_{iteration:04d}.pt';p.write_bytes(b'x');os.utime(p,(mtime,mtime))
            self.assertEqual(select_checkpoint(root,139,100).name,'iter_0140.pt')
            self.assertEqual(select_checkpoint(root,139,50).name,'iter_0139.pt')
            with self.assertRaises(ValueError):select_checkpoint(root,139,5)

if __name__=='__main__':unittest.main()
