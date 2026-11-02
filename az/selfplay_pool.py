"""Dynamically assign one-game jobs, retaining legacy per-game random seeds."""
import concurrent.futures
import queue
import re
import subprocess
import threading
from pathlib import Path


def game_jobs(workers, games, seed, iteration):
    return [(w,g,seed+iteration*100003+w*101+g)
            for w in range(workers) for g in range(games)]


def run_dynamic(binary, ports, workers, games, seed, iteration, output, options):
    jobs=queue.Queue()
    for job in game_jobs(workers,games,seed,iteration):jobs.put(job)
    stop=threading.Event()
    def consume(slot):
        plies=count=0
        while not stop.is_set():
            try:w,g,game_seed=jobs.get_nowait()
            except queue.Empty:break
            directory=Path(output)/f'worker_{w:04d}'/f'game_{g:04d}'
            command=(f'selfplay --games 1 --seed {game_seed} --out {directory} '
                     f'--eval pipe --port {ports[slot%len(ports)]} {options}\nquit\n')
            try:
                with subprocess.Popen([binary],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,text=True) as proc:
                    stdout,stderr=proc.communicate(command)
                match=re.search(r'game 0: (\d+) plies',stdout)
                if proc.returncode or not match:
                    raise RuntimeError(f'self-play job {w}/{g} failed ({proc.returncode}): {stderr[-1000:]} {stdout[-500:]}')
                count+=1;plies+=int(match.group(1))
            except BaseException:
                stop.set()
                raise
        return plies,count
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(consume,range(workers)))
    return sum(x[0] for x in results),sum(x[1] for x in results)
