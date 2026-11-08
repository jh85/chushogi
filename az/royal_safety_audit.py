#!/usr/bin/env python3
"""Retrospective sampled safety audit; does not run neural inference.

Proofs are recomputed at depth 3 / 2000 nodes. Historical proof flags and TT
states were not saved, so this cannot count original probe verdicts exactly.
Only played, visit-argmax, and >=10%-mass moves are probed. Sampling is uniform
within the position pool of 20 randomly chosen games per stratum.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import subprocess
import time

INITIAL='lfcsgekgscfl/a1b1txot1b1a/mvrhdqndhrvm/pppppppppppp/3i4i3/12/12/3I4I3/PPPPPPPPPPPP/MVRHDNQDHRVM/A1B1TOXT1B1A/LFCSGKEGSCFL b - 1'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--gate',type=Path,required=True)
    ap.add_argument('--az',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--positions-per-stratum',type=int,default=250)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    groups={
      'selfplay_softz_40600':a.data/'value_target_experiment/branches/softz/games',
      'selfplay_a0gb_40600':a.data/'value_target_experiment/branches/a0gb/games',
      'selfplay_hybrid_64600':a.data/'hybrid_44600_to64600/main/games',
      'evaluation_hybrid_64600':a.data/'hybrid_44600_to64600/main/strength/iter_000322'}
    rng=random.Random(20261108);jobs=[];selection={}
    for name,folder in groups.items():
        files=sorted(folder.rglob('game_*.json'))
        if name=='selfplay_hybrid_64600':
            files=[p for p in files if any(s.startswith('iter_') and s[5:].isdigit() and int(s[5:])>=303 for s in p.parts)]
        if not files:raise ValueError(f'No records found for {name}: {folder}')
        selected=rng.sample(files,min(20,len(files)))
        data={p:json.loads(p.read_text()) for p in selected}
        pool=[]
        for p,r in data.items():
            if 'samples' in r:
                pool.extend((p,s['ply']) for s in r['samples'])
            else:
                pool.extend((p,i) for i in range(8,len(r['moves'])))
        chosen=rng.sample(pool,min(a.positions_per_stratum,len(pool)))
        perfile=defaultdict(set)
        for p,ply in chosen:perfile[p].add(ply)
        for p,plies in perfile.items():jobs.append((name,p,data[p],plies))
        selection[name]=dict(available_games=len(files),selected_games=list(map(str,selected)),sampled_positions=len(chosen),pool_positions=len(pool))
    gate=json.loads(a.gate.read_text());jobs.append(('rated_gate',a.gate,gate,set(range(len(gate['moves_usi'])))))
    selection['rated_gate']=dict(selected_games=[str(a.gate)],sampled_positions=len(gate['moves_usi']))
    (a.out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    started=time.time()
    def audit(job):
        name,path,rec,chosen=job
        samples={s['ply']:s for s in rec.get('samples',[])}
        moves=rec.get('moves',rec.get('moves_usi'))
        initial=rec.get('initial',rec.get('initial_sfen',INITIAL))
        rows=[]
        with (a.out/f'worker_{name}_{path.stem}.log').open('w') as errors:
            proc=subprocess.Popen([a.az],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True,bufsize=1)
            def cmd(line):
                proc.stdin.write(line+'\n');proc.stdin.flush()
                reply=proc.stdout.readline().strip()
                if not reply:raise RuntimeError(f'audit engine exited at {path}: {line[:100]}')
                return reply
            try:
                assert cmd('position sfen '+initial)=='ok'
                final_ply=max(chosen)
                for ply,move in enumerate(moves):
                    if ply>final_ply:break
                    if ply not in chosen:assert cmd('audit-step '+move)=='ok';continue
                    sample=samples.get(ply)
                    policy=' '.join(f'{index}:{mass}' for index,mass in sample['policy']) if sample else ''
                    reply=cmd(f'audit-root {move} 3 2000 '+policy)
                    result=json.loads(reply.removeprefix('audit '))
                    result.update(group=name,file=str(path),ply=ply,move=move,policy_observed=sample is not None)
                    rows.append(result)
                proc.stdin.write('quit\n');proc.stdin.flush();proc.wait(timeout=10)
                if proc.returncode:raise RuntimeError('audit engine failed')
            finally:
                if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
        return rows
    results=[]
    with (a.out/'positions.jsonl').open('w') as log:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for rows in pool.map(audit,jobs):
                results.extend(rows)
                for row in rows:log.write(json.dumps(row)+'\n')
                log.flush()
    summary={}
    for group in selection:
        rows=[r for r in results if r['group']==group];n=len(rows);policies=sum(r['policy_observed'] for r in rows)
        counts=Counter()
        for r in rows:
            for key in ['played_hanging','played_reproved_loss']:
                counts[key]+=bool(r[key]);counts[key+'_with_alternative']+=bool(r[key] and r['unproved_nonhanging_alternative'])
            if r['policy_observed']:
                for key in ['argmax_hanging','argmax_reproved_loss','mass_ge_10_hanging','mass_ge_10_reproved_loss']:
                    counts[key]+=bool(r[key]);counts[key+'_with_alternative']+=bool(r[key] and r['unproved_nonhanging_alternative'])
        summary[group]=dict(positions=n,positions_with_policy=policies,counts=dict(counts),
            rates={k:v/(policies if k.startswith(('argmax','mass_')) else n) for k,v in counts.items() if (policies if k.startswith(('argmax','mass_')) else n)})
    report=dict(seconds=time.time()-started,probe_depth=3,probe_nodes=2000,summary=summary,
      limitations=['Sampled strata; not a census of all training games.',
        'Historical probe flags/TT states unavailable; re-proved loss is retrospective, not an original-search verdict.',
        'Only played, visit-argmax and individual >=10%-mass policy moves are checked; small unsafe policy entries are not counted.',
        'An alternative is a nonhanging legal move for which this bounded probe cannot prove loss; it need not be winning.',
        'Evaluation/gate records lack visit policies; only actually played moves can be audited there.',
        'Immediate hanging requires an available reply before the 1000-ply cap; deeper re-proofs are conservatively skipped near the cap.'])
    (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
