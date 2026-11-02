#!/usr/bin/env python3
"""Compare root policy temperatures and repetition search on saved game histories.

Requires an audit binary supporting `analyze az --repetition early|fourfold`.
Build separately so an active training run continues to use its original binary:
  make AZBIN=/tmp/chushogi-repetition-audit /tmp/chushogi-repetition-audit
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

from eval_match import INITIAL, Player


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def select_games(directory, candidate, opponent, count):
    games = []
    for path in sorted(directory.glob('game_*.json')):
        rec = json.loads(path.read_text())
        if (Path(rec['later']).resolve() == candidate and
                Path(rec['earlier']).resolve() == opponent and
                rec['reason'] == 'repetition'):
            games.append((path, rec))
    games.sort(key=lambda item: (len(item[1]['moves']), str(item[0])))
    if not games:
        raise ValueError('no matching repetition games')
    n = min(count, len(games))
    return [games[round(i * (len(games)-1) / max(n-1, 1))] for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--games-dir', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--opponent', type=Path, required=True)
    ap.add_argument('--az', required=True)
    ap.add_argument('--gen', default='./chushogi-gen')
    ap.add_argument('--games', type=int, default=10)
    ap.add_argument('--sims', type=int, default=256)
    ap.add_argument('--budgets', type=int, nargs='+',
                    help='compare multiple simulation budgets on the same positions')
    ap.add_argument('--temperatures', type=float, nargs='+', default=[1.0],
                    help='root policy temperatures to compare (default: 1.0)')
    ap.add_argument('--candidate-only', action='store_true',
                    help='analyze only positions where the candidate was to move')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if min(a.games, a.sims, a.workers) <= 0:
        ap.error('games, sims, and workers must be positive')
    budgets = sorted(set(a.budgets or [a.sims]))
    if min(budgets) <= 0:
        ap.error('simulation budgets must be positive')
    temperatures = sorted(set(a.temperatures))
    if any(not math.isfinite(t) or t <= 0 for t in temperatures):
        ap.error('temperatures must be finite and positive')
    candidate, opponent = a.candidate.resolve(strict=True), a.opponent.resolve(strict=True)
    selected = select_games(a.games_dir, candidate, opponent, a.games)
    a.out.mkdir(parents=True, exist_ok=False)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config['audit_binary_sha256'] = hashlib.sha256(Path(a.az).read_bytes()).hexdigest()
    (a.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    ref = Player(a.gen, 'go random')
    positions = []
    try:
        for path, rec in selected:
            ref.cmd(f'position sfen {INITIAL}', 'ok')
            keys = [' '.join(INITIAL.split()[:3])]
            for move in rec['moves']:
                ref.cmd(f'move {move}', 'ok')
                keys.append(' '.join(ref.cmd('sfen', 'sfen').split()[:3]))
            occurrences = [i for i,key in enumerate(keys) if key==keys[-1]]
            if len(occurrences) < 4:
                raise RuntimeError(f'{path}: referee did not verify fourfold repetition')
            first, second = occurrences[:2]
            cutpoints = [('before_loop',first-1), ('loop_entry',first),
                         ('before_return',second-1), ('after_return',second)]
            used = set()
            for label, ply in cutpoints:
                if ply < 8 or ply >= len(rec['moves']) or ply in used:
                    continue
                used.add(ply)
                checkpoint = candidate if ply%2==rec['color'] else opponent
                positions.append(dict(id=len(positions), game=str(path.resolve()),
                    label=label, ply=ply, initial=INITIAL, moves=rec['moves'][:ply],
                    sfen=f'{keys[ply]} {ply+1}',
                    pieces_remaining=sum(c.isalpha() for c in keys[ply].split()[0]),
                    original_move=rec['moves'][ply], checkpoint=str(checkpoint),
                    candidate_to_move=checkpoint==candidate,
                    actual_prior_occurrences=keys[:ply].count(keys[ply]),
                    terminal_position_occurrences=occurrences,
                    final_cycle_periods=[b-a for a,b in zip(occurrences,occurrences[1:])]))
    finally:
        ref.close()
    if a.candidate_only:
        positions = [p for p in positions if p['candidate_to_move']]
    if not positions:
        raise ValueError('no eligible positions selected')
    (a.out/'positions.json').write_text(json.dumps(positions,indent=2)+'\n')
    print(f'Selected {len(positions)} positions from {len(selected)} verified repetition games.',flush=True)

    servers, logs, ports = [], [], {}
    try:
        for i,path in enumerate((candidate,opponent)):
            port=free_port(); ports[str(path)]=port
            log=open(a.out/f'server_{i}.log','w');logs.append(log)
            servers.append(subprocess.Popen([sys.executable,
                str(Path(__file__).with_name('eval_server.py')),'--port',str(port),
                '--checkpoint',str(path)],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}))
        deadline=time.monotonic()+120
        for i,server in enumerate(servers):
            while 'serving' not in (a.out/f'server_{i}.log').read_text():
                if server.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError(f'eval server failed: {a.out}/server_{i}.log')
                time.sleep(.2)

        def analyze(job):
            position, sims, temperature = job
            player=Player(a.az,'',port=ports[position['checkpoint']])
            result={**position,'sims':sims,'temperature':temperature,'searches':{}}
            try:
                for mode in ('early','fourfold'):
                    player.cmd('seed 20261021','ok')
                    player.cmd(f'position sfen {INITIAL} moves '+' '.join(position['moves']),'ok')
                    t=time.monotonic()
                    data=json.loads(player.cmd(
                        f'analyze az --sims {sims} --batch 16 --repetition {mode} '
                        f'--root-policy-temp {temperature}','analysis'))
                    if not math.isclose(data.get('root_policy_temperature', float('nan')), temperature, rel_tol=1e-5):
                        raise RuntimeError('audit binary did not apply requested root temperature')
                    data['seconds']=time.monotonic()-t
                    data['moves'].sort(key=lambda m:-m['visits'])
                    data['nonrepeat_visit_fraction']=sum(m['visits'] for m in data['moves'] if m['child_prior_occurrences']==0)/data['visits']
                    data['visited_moves']=sum(m['visits']>0 for m in data['moves'])
                    data['prior_entropy']=-sum(m['prior']*math.log(m['prior']) for m in data['moves'] if m['prior']>0)
                    data['selected_move']=data['moves'][0]['move']
                    data['repetition_visit_fraction']=sum(
                        m['repetition_visits'] for m in data['moves'])/data['visits']
                    result['searches'][mode]=data
            finally:
                player.close()
            early,four=result['searches']['early'],result['searches']['fourfold']
            result['changed_move']=early['selected_move']!=four['selected_move']
            result['reproduced_original']=early['selected_move']==position['original_move']
            suffix = f'_sims_{sims:05d}' if len(budgets)>1 else ''
            if len(temperatures)>1:
                suffix += '_temp_'+format(temperature, 'g').replace('.', 'p')
            (a.out/f'position_{position["id"]:03d}{suffix}.json').write_text(json.dumps(result,indent=2)+'\n')
            print(f'position {position["id"]} ply {position["ply"]}, sims {sims}, T {temperature}: '
                  f'{early["selected_move"]} -> {four["selected_move"]}; '
                  f'repetition visits {early["repetition_visit_fraction"]:.1%} -> '
                  f'{four["repetition_visit_fraction"]:.1%}',flush=True)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            results=list(pool.map(analyze,[(p,b,t) for b in budgets for t in temperatures for p in positions]))

        def summarize(group, sims):
            candidate_results=[r for r in group if r['candidate_to_move']]
            draw_probabilities=[r['searches']['early']['root_wdl'][1]
                                for r in candidate_results]
            return dict(games=len(selected),positions=len(group),sims=sims,
                temperature=group[0]['temperature'],
                selected_repeat_moves={mode:sum(r['searches'][mode]['moves'][0]['child_prior_occurrences']>0 for r in group) for mode in ('early','fourfold')},
                mean_nonrepeat_visit_fraction={mode:statistics.mean(r['searches'][mode]['nonrepeat_visit_fraction'] for r in group) for mode in ('early','fourfold')},
                mean_prior_entropy={mode:statistics.mean(r['searches'][mode]['prior_entropy'] for r in group) for mode in ('early','fourfold')},
                mean_visited_moves={mode:statistics.mean(r['searches'][mode]['visited_moves'] for r in group) for mode in ('early','fourfold')},
                reproduced_original=sum(r['reproduced_original'] for r in group),
                changed_moves=sum(r['changed_move'] for r in group),
                candidate_positions=len(candidate_results),
                candidate_changed_moves=sum(r['changed_move'] for r in candidate_results),
                candidate_draw_probability_median=(statistics.median(draw_probabilities)
                    if draw_probabilities else None),
                candidate_draw_probability_above_99pct=sum(p>.99 for p in draw_probabilities),
                mean_repetition_visit_fraction={mode:sum(r['searches'][mode]['repetition_visit_fraction']
                    for r in group)/len(group) for mode in ('early','fourfold')},
                cases=[dict(id=r['id'],ply=r['ply'],label=r['label'],candidate_to_move=r['candidate_to_move'],
                            early_move=r['searches']['early']['selected_move'],
                            fourfold_move=r['searches']['fourfold']['selected_move'],
                            changed=r['changed_move'],reproduced_original=r['reproduced_original']) for r in group])

        conditions={f'{b}@{t:g}':summarize(
            [r for r in results if r['sims']==b and r['temperature']==t],b)
            for b in budgets for t in temperatures}
        if len(temperatures)==1:
            by_budget={str(b):conditions[f'{b}@{temperatures[0]:g}'] for b in budgets}
            summary = by_budget[str(budgets[0])] if len(budgets)==1 else dict(
                games=len(selected), positions=len(positions), budgets=by_budget)
        else:
            comparisons=[]
            baseline=temperatures[0]
            for b in budgets:
                for t in temperatures[1:]:
                    pairs=[(next(r for r in results if r['id']==p['id'] and r['sims']==b and r['temperature']==baseline),
                            next(r for r in results if r['id']==p['id'] and r['sims']==b and r['temperature']==t)) for p in positions]
                    comparisons.append(dict(sims=b,baseline=baseline,temperature=t,
                        repeat_to_nonrepeat={mode:sum(x['searches'][mode]['moves'][0]['child_prior_occurrences']>0 and y['searches'][mode]['moves'][0]['child_prior_occurrences']==0 for x,y in pairs) for mode in ('early','fourfold')},
                        nonrepeat_to_repeat={mode:sum(x['searches'][mode]['moves'][0]['child_prior_occurrences']==0 and y['searches'][mode]['moves'][0]['child_prior_occurrences']>0 for x,y in pairs) for mode in ('early','fourfold')},
                        changed_moves={mode:sum(x['searches'][mode]['selected_move']!=y['searches'][mode]['selected_move'] for x,y in pairs) for mode in ('early','fourfold')}))
            summary=dict(games=len(selected),positions=len(positions),conditions=conditions,
                         temperature_comparisons=comparisons)
        summary['limitations']=['Selected positions target the final repeated position, not a random position sample.',
            'Changed choices do not establish improved playing strength or winning alternatives.',
            'Actual game adjudication and production training settings were not changed.']
        (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({b:{k:v for k,v in s.items() if k!='cases'}
                          for b,s in conditions.items()},indent=2))
    finally:
        for server in servers:server.terminate()
        for server in servers:server.wait(timeout=10)
        for log in logs:log.close()


if __name__=='__main__':
    main()
