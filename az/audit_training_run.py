#!/usr/bin/env python3
"""Audit optimizer continuity and sampled self-play distribution without changing training."""
import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import numpy as np
import torch


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--start',type=int,default=80)
    ap.add_argument('--end',type=int,default=139)
    ap.add_argument('--sample-games',type=int,default=4)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((a.run_dir/'loop_config.json').read_text())
    iterations=[]
    for it in range(a.start,a.end+1):
        log=a.run_dir/f'train_{it:06d}.log'
        text=log.read_text() if log.exists() else ''
        logs=[dict(step=int(step),loss=float(loss),policy=float(p),value=float(v),lr=float(lr))
              for step,loss,p,v,lr in re.findall(r'step (\d+) loss ([\d.]+) \(p ([\d.]+) v ([\d.]+)\) lr ([\deE.+-]+)',text)]
        files=sorted((a.run_dir/'games'/f'iter_{it:06d}').rglob('game_*.json'))
        n=min(a.sample_games,len(files))
        selected=[files[round(i*(len(files)-1)/max(n-1,1))] for i in range(n)]
        games=[json.loads(p.read_text()) for p in selected]
        sample_data=[g['samples'][i] for g in games for i in [len(g['samples'])//4,len(g['samples'])//2,3*len(g['samples'])//4] if g['samples']]
        entropy=[-sum(prob*np.log(prob) for _,prob in s['policy'] if prob>0) for s in sample_data]
        iterations.append(dict(iteration=it,files=len(files),sampled_games=len(games),
            sampled_endings=dict(Counter(g['reason'] for g in games)),
            sampled_mean_plies=statistics.mean(len(g['moves']) for g in games) if games else None,
            sampled_mean_policy_entropy=statistics.mean(entropy) if entropy else None,
            sampled_rootq_minus_outcome_mae=statistics.mean(abs(s['rootq']-s['z']) for s in sample_data) if sample_data else None,
            buffer_size=int(re.search(r'buffer: (\d+)',text)[1]) if 'buffer:' in text else None,
            optimizer_restored='restored optimizer state' in text,logs=logs))
    checkpoint_stats=[];previous=None
    for it in [80,89,119,129,139]:
        path=a.run_dir/'ckpt'/f'iter_{it:04d}.pt'
        state=torch.load(path,map_location='cpu',weights_only=True)
        model=state['model'];norm=sum(float(x.float().square().sum()) for x in model.values())**.5
        group=state['optimizer']['param_groups'][0]
        steps=sorted({int(v['step']) for v in state['optimizer']['state'].values()})
        entry=dict(iteration=it,step=state['step'],optimizer_steps=steps,
                   optimizer_lr=group['lr'],betas=group['betas'],weight_decay=group['weight_decay'],
                   parameter_l2=norm,train_config=state.get('config'))
        if previous is not None:
            pn=sum(float(x.float().square().sum()) for x in previous.values())**.5
            delta=sum(float((model[k].float()-previous[k].float()).square().sum()) for k in model)**.5
            entry['relative_parameter_change_from_previous_selected_checkpoint']=delta/pn
        checkpoint_stats.append(entry);previous=model
    report=dict(config=cfg,iterations=iterations,checkpoint_stats=checkpoint_stats,
        limitations=['Self-play distributions are sampled, not all games.',
          'Training losses are from changing replay distributions, not a fixed validation set.',
          'Observed correlations do not establish a cause of regression.'])
    (a.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(iterations=len(iterations),all_optimizer_restored=all(r['optimizer_restored'] for r in iterations),
        buffer_sizes=sorted({r['buffer_size'] for r in iterations if r['buffer_size'] is not None}),
        checkpoint_stats=[{k:v for k,v in r.items() if k!='train_config'} for r in checkpoint_stats]),indent=2))

if __name__=='__main__':main()
