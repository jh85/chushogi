"""AdamW and native PyTorch Muon/AdamW with explicit checkpoint migration."""
import copy
import math
import torch


def partition_parameters(net):
    muon, adamw = [], []
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        (muon if name.startswith('trunk.') and parameter.ndim == 2 else adamw).append((name, parameter))
    return muon, adamw


class Optimizers:
    def __init__(self, net, kind='adamw', lr=.001, weight_decay=.01,
                 muon_lr_scale=1., muon_warmup_steps=50):
        if kind not in ('adamw', 'muon_adamw'):
            raise ValueError('unknown optimizer')
        if not math.isfinite(muon_lr_scale) or muon_lr_scale <= 0 or muon_warmup_steps < 0:
            raise ValueError('invalid Muon learning-rate scale or warmup')
        self.kind = kind
        self.names = [name for name, p in net.named_parameters() if p.requires_grad]
        self.muon_updates = 0
        self.muon_lr_scale = muon_lr_scale
        self.muon_warmup_steps = muon_warmup_steps
        self.weight_decay = weight_decay
        self.muon = None
        self.migration = 'fresh'
        matrices, remaining = partition_parameters(net)
        if kind == 'adamw':
            self.adamw = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
            self.adamw_names = self.names
            self.muon_names = []
        else:
            if not hasattr(torch.optim, 'Muon'):
                raise RuntimeError('this PyTorch does not provide torch.optim.Muon')
            if not matrices or not remaining:
                raise ValueError('hybrid optimizer requires both parameter partitions')
            self.muon_names = [n for n, _ in matrices]
            self.adamw_names = [n for n, _ in remaining]
            self.muon = torch.optim.Muon([p for _, p in matrices], lr=lr*muon_lr_scale,
                weight_decay=weight_decay, momentum=.95, nesterov=True, ns_steps=5,
                adjust_lr_fn='match_rms_adamw')
            self.adamw = torch.optim.AdamW([p for _, p in remaining], lr=lr, weight_decay=weight_decay)

    def load_checkpoint(self, checkpoint):
        saved = checkpoint.get('optimizer')
        if saved is None:
            return
        hybrid = saved.get('kind') == 'muon_adamw'
        if hybrid:
            if self.kind != 'muon_adamw':
                raise ValueError('cannot restore hybrid state as AdamW; use --optimizer muon_adamw')
            if saved.get('version') != 1 or saved['muon_names'] != self.muon_names or saved['adamw_names'] != self.adamw_names:
                raise ValueError('hybrid checkpoint parameter partition/version mismatch')
            if (saved.get('muon_lr_scale', self.muon_lr_scale) != self.muon_lr_scale or
                    saved.get('muon_warmup_steps', self.muon_warmup_steps) != self.muon_warmup_steps):
                raise ValueError('resume with the saved Muon learning-rate scale and warmup')
            self.muon.load_state_dict(saved['muon'])
            self.adamw.load_state_dict(saved['adamw'])
            self.muon_updates = int(saved['muon_updates'])
            self.migration = 'restored_muon_and_adamw'
        elif self.kind == 'adamw':
            if checkpoint.get('optimizer_param_names', self.names) != self.names:
                raise ValueError('AdamW checkpoint parameter ordering mismatch')
            self.adamw.load_state_dict(saved)
            self.migration = 'restored_adamw'
        else:
            # Legacy checkpoints used net.parameters() in a single AdamW group.
            if len(saved['param_groups']) != 1:
                raise ValueError('legacy AdamW migration requires one parameter group')
            source_names = checkpoint.get('optimizer_param_names', self.names)
            ids = saved['param_groups'][0]['params']
            if source_names != self.names or len(ids) != len(source_names):
                raise ValueError('legacy AdamW parameter ordering mismatch')
            source = dict(zip(source_names, ids))
            target = self.adamw.state_dict()
            destination_ids = target['param_groups'][0]['params']
            for name, destination in zip(self.adamw_names, destination_ids):
                if source[name] in saved['state']:
                    target['state'][destination] = copy.deepcopy(saved['state'][source[name]])
            group = copy.deepcopy(saved['param_groups'][0])
            group['params'] = destination_ids
            group.pop('param_names', None)
            target['param_groups'] = [group]
            self.adamw.load_state_dict(target)
            self.migration = 'preserved_adamw_auxiliary_state_fresh_muon_momentum'
        for group in self.adamw.param_groups:
            group['weight_decay'] = self.weight_decay
        if self.muon is not None:
            for group in self.muon.param_groups:
                group.update(weight_decay=self.weight_decay, momentum=.95, nesterov=True,
                             ns_steps=5, adjust_lr_fn='match_rms_adamw')

    def set_lr(self, lr):
        for group in self.adamw.param_groups:
            group['lr'] = lr
        if self.muon is not None:
            ramp = min(1., (self.muon_updates+1)/max(1, self.muon_warmup_steps))
            for group in self.muon.param_groups:
                group['lr'] = lr*self.muon_lr_scale*ramp

    def zero_grad(self, set_to_none=True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.adamw.step()
        if self.muon is not None:
            self.muon.step()
            self.muon_updates += 1

    def state_dict(self):
        if self.muon is None:
            return self.adamw.state_dict()
        return dict(version=1, kind='muon_adamw', muon=self.muon.state_dict(),
                    adamw=self.adamw.state_dict(), muon_names=self.muon_names,
                    adamw_names=self.adamw_names, muon_updates=self.muon_updates,
                    muon_lr_scale=self.muon_lr_scale, muon_warmup_steps=self.muon_warmup_steps)
