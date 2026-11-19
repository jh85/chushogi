"""Actual USI regression, including mate override and missing evaluator fallback."""
from pathlib import Path
import queue
import subprocess
import threading
import unittest

INITIAL='6k1s3/6e5/8c3/5+O4+M1/2p9/12/12/2P9/F1M7T1/1S8S1/6E5/6K5 b - 447'
CYCLE=['7d8c','6a5b','8c7d','5b6a']


class UsiRepetitionTest(unittest.TestCase):
    def run_search(self,guard=True,random=True,white=False,sampling=16):
        process=subprocess.Popen([str(Path('chushogi-az').resolve())],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        output=queue.Queue()
        def drain():
            for line in process.stdout:output.put(line.strip())
        thread=threading.Thread(target=drain,daemon=True);thread.start()
        try:
            initial=INITIAL;moves=(CYCLE*3)[:10];bad='8c7d'
            if white:
                import sys
                sys.path.insert(0,str(Path('az').resolve()))
                from encode import Position,flip_usi_move
                # Independently mirror the fixture with the Python encoder.
                from encode import ROLES,_SFEN_PLAIN,_SFEN_PROMOTED
                position=Position(INITIAL).flipped()
                plain={ROLES.index(v):k for k,v in _SFEN_PLAIN.items()}
                promoted={ROLES.index(v):'+'+k for k,v in _SFEN_PROMOTED.items()}
                ranks=[]
                for y in range(12):
                    row='';empty=0
                    for x in reversed(range(12)):
                        piece=position.board[y*12+x]
                        if piece is None:empty+=1;continue
                        if empty:row+=str(empty);empty=0
                        role,color=piece;symbol=plain.get(role,promoted.get(role))
                        row+=symbol.upper() if color==0 else symbol
                    if empty:row+=str(empty)
                    ranks.append(row)
                initial='/'.join(ranks)+' w - 447'
                moves=[flip_usi_move(m) for m in moves];bad=flip_usi_move(bad)
            commands=['usi','setoption name Sims value 16','setoption name Batch value 4',
                f'setoption name OpeningSamplingPlies value {sampling}',
                'setoption name Seed value 1',
                'setoption name PerpetualCheckGuard value '+('true' if guard else 'false'),
                'setoption name Eval value '+('random' if random else 'pipe'),
                'setoption name EvalPort value 1','usinewgame',
                'position sfen '+initial+' moves '+' '.join(moves),'go nodes 16']
            process.stdin.write('\n'.join(commands)+'\n');process.stdin.flush()
            lines=[]
            while True:
                line=output.get(timeout=20);lines.append(line)
                if line.startswith('bestmove '):return line.split()[1],lines,bad
        finally:
            if process.poll() is None:
                process.stdin.write('quit\n');process.stdin.flush()
                process.wait(timeout=20)
            process.stdin.close();process.stdout.close();process.stderr.close();thread.join(timeout=1)

    def test_legacy_mate_override_reproduces_the_losing_move(self):
        best,lines,bad=self.run_search(guard=False)
        self.assertEqual(best,bad)

    def test_guard_protects_black_with_mate_enabled(self):
        best,lines,bad=self.run_search()
        self.assertNotEqual(best,bad)
        self.assertTrue(any('perpetual-check guard: excluded '+bad in line for line in lines))

    def test_guard_protects_white_through_canonical_conversion(self):
        best,lines,bad=self.run_search(white=True)
        self.assertNotEqual(best,bad)
        self.assertTrue(any('perpetual-check guard: excluded '+bad in line for line in lines))

    def test_no_evaluator_fallback_cannot_bypass_the_guard(self):
        best,lines,bad=self.run_search(random=False)
        self.assertNotEqual(best,bad)
        self.assertTrue(any('no eval server' in line for line in lines))

    def test_sampling_cannot_bypass_the_perpetual_guard(self):
        for white in [False, True]:
            best,lines,bad=self.run_search(white=white,sampling=1000)
            self.assertNotEqual(best,bad)
            self.assertTrue(any('perpetual-check guard: excluded '+bad in line for line in lines))
            self.assertTrue(any('opening sampling:' in line for line in lines))


if __name__=='__main__':unittest.main()
