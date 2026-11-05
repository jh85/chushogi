"""Optimizer partition, migration and exact continuation checks."""
import copy
import io
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'az'))
from model import ChuNet
from optimizers import Optimizers,partition_parameters

torch.set_num_threads(1)

class OptimizerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.net=ChuNet(16,1,4)
        self.x=torch.randn(2,82,12,12)
    def update(self,net,opt):
        opt.zero_grad(set_to_none=True)
        p,v=net(self.x);(p.square().mean()+v.square().mean()).backward();opt.step()
    def legacy(self):
        opt=torch.optim.AdamW(self.net.parameters(),lr=.001,weight_decay=.01)
        self.update(self.net,opt)
        return opt,dict(optimizer=copy.deepcopy(opt.state_dict()))
    def test_partition_and_legacy_auxiliary_moments(self):
        old,saved=self.legacy()
        hybrid=Optimizers(self.net,'muon_adamw');hybrid.load_checkpoint(saved)
        matrices,aux=partition_parameters(self.net)
        self.assertEqual(len(matrices),4)
        self.assertEqual(len(hybrid.muon.state),0)
        for _,p in aux:
            for k in ['step','exp_avg','exp_avg_sq']:
                torch.testing.assert_close(hybrid.adamw.state[p][k],old.state[p][k],rtol=0,atol=0)
        self.assertTrue(all(n.startswith('trunk.') and p.ndim==2 for n,p in matrices))
        self.assertTrue(all(not (n.startswith('trunk.') and p.ndim==2) for n,p in aux))
        self.assertTrue(hybrid.muon.param_groups[0]['adjust_lr_fn']=='match_rms_adamw')
        hybrid.set_lr(.001)
        self.assertAlmostEqual(hybrid.muon.param_groups[0]['lr'],.001/50)
        self.update(self.net,hybrid)
        self.assertEqual(hybrid.muon_updates,1)
        self.assertEqual(len(hybrid.muon.state),4)
    def test_adamw_continuation_is_exact(self):
        old,saved=self.legacy()
        clone=copy.deepcopy(self.net)
        new=Optimizers(clone);new.load_checkpoint(saved);new.set_lr(.001)
        self.update(self.net,old);self.update(clone,new)
        for a,b in zip(self.net.parameters(),clone.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_hybrid_save_resume_is_exact_including_warmup(self):
        _,saved=self.legacy()
        first=Optimizers(self.net,'muon_adamw',muon_lr_scale=.3)
        first.load_checkpoint(saved)
        for _ in range(3):first.set_lr(.001);self.update(self.net,first)
        clone=copy.deepcopy(self.net)
        second=Optimizers(clone,'muon_adamw',muon_lr_scale=.3)
        serialized=io.BytesIO()
        torch.save(dict(optimizer=first.state_dict()),serialized)
        serialized.seek(0)
        second.load_checkpoint(torch.load(serialized,weights_only=True))
        self.assertEqual(second.muon_updates,3)
        for _ in range(2):
            first.set_lr(.001);second.set_lr(.001)
            self.assertEqual(first.muon.param_groups[0]['lr'],second.muon.param_groups[0]['lr'])
            self.update(self.net,first);self.update(clone,second)
        for a,b in zip(self.net.parameters(),clone.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'saved Muon learning-rate'):
            Optimizers(clone,'muon_adamw',muon_lr_scale=3.).load_checkpoint(dict(optimizer=first.state_dict()))
        with self.assertRaisesRegex(ValueError,'hybrid state'):
            Optimizers(clone).load_checkpoint(dict(optimizer=first.state_dict()))
    def test_incompatible_legacy_order_rejected(self):
        _,saved=self.legacy()
        saved['optimizer_param_names']=['wrong']*len(list(self.net.parameters()))
        with self.assertRaisesRegex(ValueError,'ordering mismatch'):
            Optimizers(self.net,'muon_adamw').load_checkpoint(saved)

if __name__=='__main__':unittest.main()
