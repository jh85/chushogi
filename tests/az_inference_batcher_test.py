import concurrent.futures
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'az'))
from inference_batcher import InferenceBatcher

class BatcherTest(unittest.TestCase):
    def test_order_bounds_and_different_request_sizes(self):
        calls=[]
        def evaluate(raw, progress):
            calls.append((raw,progress.copy()))
            return progress[:,None].repeat(2,axis=1),progress[:,None]
        batcher=InferenceBatcher(evaluate, max_batch=8, wait_ms=20)
        try:
            futures=[batcher.submit(bytes([i])*n,np.full(n,i)) for i,n in enumerate([3,2,4,1])]
            for i,(future,n) in enumerate(zip(futures,[3,2,4,1])):
                p,v=future.result(timeout=2)
                np.testing.assert_array_equal(p,np.full((n,2),i))
                np.testing.assert_array_equal(v,np.full((n,1),i))
            self.assertTrue(all(len(prog)<=8 for _,prog in calls))
            self.assertEqual(b''.join(raw for raw,_ in calls),bytes(3)+bytes([1])*2+bytes([2])*4+bytes([3]))
            self.assertLess(len(calls),4)
        finally:batcher.close()
        with self.assertRaises(RuntimeError):batcher.submit(b'x',np.ones(1))

    def test_ragged_legal_requests_and_full_mode_boundary(self):
        calls=[]
        def evaluate(raw, progress, indices=None):
            calls.append(indices)
            if indices is None:return progress[:,None],progress[:,None]
            out=np.zeros((len(indices),max(map(len,indices))))
            for i,row in enumerate(indices):out[i,:len(row)]=np.asarray(row)+progress[i]
            return out,progress[:,None]
        batcher=InferenceBatcher(evaluate,128,20)
        try:
            a=batcher.submit(b'a',np.array([10]),[[4,1]])
            b=batcher.submit(b'b',np.array([20,30]),[[],[2,5,0]])
            c=batcher.submit(b'c',np.array([40]))
            d=batcher.submit(b'd',np.array([50]),[[7]])
            np.testing.assert_array_equal(a.result(timeout=2)[0][0,:2],[14,11])
            np.testing.assert_array_equal(b.result(timeout=2)[0][1,:3],[32,35,30])
            self.assertEqual(c.result(timeout=2)[0][0,0],40)
            self.assertEqual(d.result(timeout=2)[0][0,0],57)
            self.assertEqual(calls,[[[4,1],[],[2,5,0]],None,[[7]]])
        finally:batcher.close()

    def test_failure_propagates_and_worker_recovers(self):
        def evaluate(raw,progress):
            if progress[0]<0:raise ValueError('bad inference')
            return progress[:,None],progress[:,None]
        batcher=InferenceBatcher(evaluate,8,0)
        try:
            with self.assertRaisesRegex(ValueError,'bad inference'):
                batcher.submit(b'x',np.array([-1])).result(timeout=2)
            self.assertEqual(batcher.submit(b'x',np.array([2])).result(timeout=2)[0][0,0],2)
        finally:batcher.close()

    def test_close_drains_queued_requests(self):
        batcher=InferenceBatcher(lambda raw,p:(p[:,None],p[:,None]),8,20)
        futures=[batcher.submit(b'x',np.array([i])) for i in range(30)]
        batcher.close()
        self.assertTrue(all(f.done() and f.exception() is None for f in futures))

if __name__=='__main__':unittest.main()
