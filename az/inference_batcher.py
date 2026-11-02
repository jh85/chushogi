"""Combine independent inference requests without changing per-game search batches."""
from concurrent.futures import Future
import queue
import threading
import time
import numpy as np


class InferenceBatcher:
    def __init__(self, evaluate, max_batch=128, wait_ms=1.0):
        if max_batch < 1 or wait_ms < 0:
            raise ValueError('batch size must be positive and wait nonnegative')
        self.evaluate = evaluate
        self.max_batch = max_batch
        self.wait_seconds = wait_ms / 1000
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.closed = False
        self.stats = dict(requests=0, positions=0, batches=0, largest_batch=0)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, packed, progress, indices=None):
        future = Future()
        with self.lock:
            if self.closed:
                raise RuntimeError('batcher closed')
            self.queue.put((packed, progress, future, time.monotonic(), indices))
        return future

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.queue.put(None)
        self.thread.join()

    def _run(self):
        pending = None
        closing = False
        while True:
            first = pending if pending is not None else self.queue.get()
            pending = None
            if first is None:
                return
            requests = [first]
            size = len(first[1])
            deadline = first[3] + self.wait_seconds
            while size < self.max_batch:
                try:
                    item = self.queue.get(timeout=max(0, deadline-time.monotonic()))
                except queue.Empty:
                    break
                if item is None:
                    closing = True
                    break
                if size + len(item[1]) > self.max_batch or (item[4] is None) != (first[4] is None):
                    pending = item
                    break
                requests.append(item)
                size += len(item[1])
            try:
                packed = b''.join(r[0] for r in requests)
                progress = np.concatenate([r[1] for r in requests])
                if first[4] is None:
                    policy, wdl = self.evaluate(packed, progress)
                else:
                    indices = [row for r in requests for row in r[4]]
                    policy, wdl = self.evaluate(packed, progress, indices)
                if len(policy) != size or len(wdl) != size:
                    raise ValueError('inference returned incorrect batch size')
                offset = 0
                for _, prog, future, _, _ in requests:
                    end = offset + len(prog)
                    future.set_result((policy[offset:end], wdl[offset:end]))
                    offset = end
                self.stats['requests'] += len(requests)
                self.stats['positions'] += size
                self.stats['batches'] += 1
                self.stats['largest_batch'] = max(self.stats['largest_batch'], size)
            except Exception as error:
                for _, _, future, _, _ in requests:
                    if not future.done():
                        future.set_exception(error)
            if closing:
                return
