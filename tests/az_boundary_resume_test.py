"""Exercise a boundary handover without launching real training or killing jobs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parent.parent

class BoundaryTest(unittest.TestCase):
    def test_handover_preserves_end_iteration_and_waits_for_clean_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';(run/'ckpt').mkdir(parents=True)
            for name in ['iter_0003.pt','current.pt']:(run/'ckpt'/name).write_bytes(b'complete-checkpoint')
            (run/'loop.log').write_text('working\n')
            engine=root/'engine';engine.write_bytes(b'engine')
            wrapper=root/'capture'
            wrapper.write_text('#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n'+
                               f'Path({str(root/"launched.json")!r}).write_text(json.dumps(sys.argv))\n')
            wrapper.chmod(0o755)
            cfg=dict(python=str(wrapper),stop_file=str(run/'STOP'),run_dir=str(run),workers=44,games=2,
                     az=str(engine),iterations=100,resume=True,dynamic_games=False)
            (run/'loop_config.json').write_text(json.dumps(cfg))
            old_script=root/'run_loop.py'
            old_script.write_text('import sys,time\nfrom pathlib import Path\nr=Path(sys.argv[1])\n'+
                                  'while not (r/"STOP").exists():time.sleep(.01)\n'+
                                  '(r/"loop.log").write_text("[loop] finished\\n")\n')
            old=subprocess.Popen([sys.executable,str(old_script),str(run)])
            try:
                subprocess.run([sys.executable,str(ROOT/'az/resume_at_boundary.py'),'--run-dir',str(run),
                    '--old-pid',str(old.pid),'--az',str(engine),'--final-iteration','9'],check=True,timeout=20)
                old.wait(timeout=2)
                deadline=time.monotonic()+2
                while not (root/'launched.json').exists() and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue((root/'launched.json').exists())
                state=json.loads((run/'throughput_handover.json').read_text())
                self.assertEqual(state['status'],'resumed')
                self.assertEqual(state['remaining_iterations'],6)
                self.assertFalse((run/'STOP').exists())
                self.assertEqual((run/'ckpt/current.pt').read_bytes(),b'complete-checkpoint')
                command=state['command']
                self.assertIn('--dynamic-games',command)
                self.assertEqual(command[command.index('--eval-max-batch')+1],'128')
                self.assertEqual(command[command.index('--iterations')+1],'6')
            finally:
                if old.poll() is None:old.terminate();old.wait()

if __name__=='__main__':unittest.main()
