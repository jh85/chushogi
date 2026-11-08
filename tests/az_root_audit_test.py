import json
from pathlib import Path
import subprocess
import unittest
from az_blunder_test import SENTE,mirror_sfen,mirror_move

ROOT=Path(__file__).resolve().parents[1]

class AuditTest(unittest.TestCase):
    def audit(self,sfen,played):
        command=f'position sfen {sfen}\naudit-root {played} 3 2000\nquit\n'
        r=subprocess.run([str(ROOT/'chushogi-az')],input=command,text=True,capture_output=True,check=True,timeout=10)
        return json.loads(next(s[6:] for s in r.stdout.splitlines() if s.startswith('audit ')))
    def test_known_gate_blunder_and_canonical_flip(self):
        for pos,played in [(SENTE,'4j5i'),(mirror_sfen(SENTE),mirror_move('4j5i'))]:
            r=self.audit(pos,played)
            self.assertTrue(r['played_hanging'])
            self.assertTrue(r['played_reproved_loss'])
            self.assertTrue(r['unproved_nonhanging_alternative'])
    def test_no_future_reply_after_ply_cap(self):
        r=self.audit(SENTE.replace('- 319','- 1000'),'4j5i')
        self.assertFalse(r['played_hanging'])
        self.assertFalse(r['played_reproved_loss'])

if __name__=='__main__':unittest.main()
