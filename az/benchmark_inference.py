#!/usr/bin/env python3
"""Benchmark framed inference latency/concurrency against a frozen checkpoint.

Launches its own server; never reloads or stops an existing training server.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PACKED = 1458
POLICY = 50688


def read_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        block = sock.recv(size - len(data))
        if not block:
            raise ConnectionError('server closed')
        data.extend(block)
    return data


def request(sock, n):
    sock.sendall(struct.pack('<I', n) + bytes(n*PACKED) + bytes(n*2))
    count, = struct.unpack('<I', read_exact(sock, 4))
    if count != n:
        raise ValueError('response count mismatch')
    return bytes(read_exact(sock, n*(POLICY+3)*2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--server-script', default=str(ROOT/'az/eval_server.py'))
    ap.add_argument('--clients', type=int, nargs='+', default=[1,16])
    ap.add_argument('--sizes', type=int, nargs='+', default=[1,16])
    ap.add_argument('--requests', type=int, default=12)
    ap.add_argument('--nodelay', action='store_true')
    ap.add_argument('--max-batch', type=int, default=0)
    ap.add_argument('--batch-wait-ms', type=float, default=1)
    ap.add_argument('--server-nodelay', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if min(*a.clients,*a.sizes,a.requests)<=0:
        ap.error('counts must be positive')
    a.out.mkdir(parents=True,exist_ok=False)
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    cmd=[sys.executable,a.server_script,'--checkpoint',a.checkpoint,'--port',str(port)]
    if a.max_batch:cmd+=['--max-batch',str(a.max_batch),'--batch-wait-ms',str(a.batch_wait_ms)]
    if a.server_nodelay:cmd+=['--tcp-nodelay']
    with open(a.out/'server.log','w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(ROOT/'az')})
    results=[]
    try:
        deadline=time.monotonic()+120
        while 'serving' not in (a.out/'server.log').read_text():
            if proc.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError((a.out/'server.log').read_text())
            time.sleep(.1)
        # Synchronize the measured section after per-client warmup.
        import threading
        for size in a.sizes:
            for clients in a.clients:
                barrier=threading.Barrier(clients)
                def worker(_):
                    with socket.create_connection(('127.0.0.1',port),timeout=60) as sock:
                        if a.nodelay:sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                        for _ in range(2):request(sock,size)
                        barrier.wait(timeout=60)
                        times=[];start=time.perf_counter()
                        for _ in range(a.requests):
                            t=time.perf_counter();data=request(sock,size);times.append(time.perf_counter()-t)
                        elapsed=time.perf_counter()-start
                        return times,elapsed,hashlib.sha256(data).hexdigest()
                with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
                    measurements=list(pool.map(worker,range(clients)))
                latencies=[v*1000 for times,_,_ in measurements for v in times]
                elapsed=max(m[1] for m in measurements)
                result=dict(size=size,clients=clients,requests_per_client=a.requests,
                    positions_per_second=clients*a.requests*size/elapsed,
                    latency_ms_p50=float(np.median(latencies)),latency_ms_p95=float(np.percentile(latencies,95)),
                    output_hashes=sorted(set(m[2] for m in measurements)))
                results.append(result);print(json.dumps(result),flush=True)
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        config['checkpoint_sha256']=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
        (a.out/'results.json').write_text(json.dumps(dict(config=config,results=results),indent=2)+'\n')
    finally:
        proc.terminate();proc.wait(timeout=10)


if __name__=='__main__':main()
