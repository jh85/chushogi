#!/usr/bin/env python3
"""Compare value calibration on the same previously unseen tournament histories.

Outcomes belong to the played policies, not an oracle of optimal position value.
Move-limit draws are marked separately. This is diagnostic, not causal evidence.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
import encode as E
import model as M
from eval_match import INITIAL,Player


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tournament',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--matchup',default='4_3')
    ap.add_argument('--games',type=int,default=100)
    ap.add_argument('--gen',default='./chushogi-gen')
    a=ap.parse_args();cfg=json.loads((a.tournament/'config.json').read_text())
    paths=cfg['unique_checkpoints']
    rows=[json.loads(p.read_text()) for p in sorted(a.tournament.glob(f'game_{a.matchup}_*.json'))[:a.games]]
    if a.games<=0 or len(rows)!=a.games:raise ValueError('requested number of match games is not available')
    ref=Player(a.gen,'go random');planes=[];targets=[];cases=[]
    try:
        for rec in rows:
            cuts={max(8,int(len(rec['moves'])*f)) for f in [.25,.5,.75]}
            cuts={n for n in cuts if n<len(rec['moves'])}
            if not cuts:continue
            ref.cmd('position sfen '+INITIAL,'ok');seen=Counter()
            for ply in range(max(cuts)+1):
                sfen=ref.cmd('sfen','sfen');key=' '.join(sfen.split()[:3])
                if ply in cuts:
                    planes.append(E.encode_position(E.Position(sfen),rep_count=seen[key],ply=ply))
                    sente_score=rec['score'] if rec['color']==0 else 1-rec['score']
                    score=sente_score if sfen.split()[1]=='b' else 1-sente_score
                    targets.append([1.,0.,0.] if score==1 else [0.,0.,1.] if score==0 else [0.,1.,0.])
                    cases.append(dict(pair=rec['pair'],color=rec['color'],ply=ply,reason=rec['reason']))
                seen[key]+=1
                if ply<max(cuts):ref.cmd('move '+rec['moves'][ply],'ok')
    finally:ref.close()
    inputs=np.stack(planes);target=np.array(targets,dtype=np.float32)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    results=[]
    for path in paths:
        state=torch.load(path,map_location='cpu',weights_only=True)
        net=M.ChuNet().to(device);net.load_state_dict(state['model']);net.eval()
        probabilities=[]
        with torch.no_grad():
            for start in range(0,len(inputs),16):
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                    _,wdl=net(torch.from_numpy(inputs[start:start+16]).to(device))
                probabilities.extend(torch.softmax(wdl.float(),dim=-1).cpu().numpy())
        prob=np.stack(probabilities);squared=((prob-target)**2).sum(axis=1)
        nll=-(target*np.log(np.clip(prob,1e-8,1))).sum(axis=1)
        decisive=np.array([c['reason']!='moveLimit' for c in cases])
        results.append(dict(checkpoint=path,step=state['step'],samples=len(target),
            wdl_brier=float(squared.mean()),wdl_nll=float(nll.mean()),
            excluding_move_limit_brier=float(squared[decisive].mean()),
            excluding_move_limit_nll=float(nll[decisive].mean())))
        del net
    report=dict(results=results,cases=cases,limitations=[
        'Multiple positions from each game are correlated.',
        'Outcomes describe played policies, not optimal position values.',
        'These final tournament games were not in the training replay buffer.',
        'Calibration differences alone cannot establish why playing strength changed.'])
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
