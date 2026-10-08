#!/usr/bin/env python3
"""AZ trainer: trains the network on self-play records from the buffer.

Loss = soft cross-entropy on the visit-count policy target
     + value_weight * cross-entropy on the WDL value target.

Publishes checkpoints atomically (write tmp + rename) so the eval server's
--watch reload picks them up mid-run.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import buffer as B
import model as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="checkpoint path to publish")
    ap.add_argument("--init-from", default="")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--lr-decay-steps", type=int, default=0,
                    help="total-step horizon for the cosine decay to the "
                         "1e-4 floor; 0 = decay over this run's --steps "
                         "(degenerates to constant 1e-4 once step0 > warmup)")
    ap.add_argument("--value-target",
                    choices=["z", "softz", "a0gb", "blend"], default="blend")
    ap.add_argument("--lam", type=float, default=0.5,
                    help="search-Q share of the blended target")
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--max-positions", type=int, default=1_000_000)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--step0", type=int, default=0, help="starting step count")
    ap.add_argument("--val-games-dir", default="",
                    help="held-out games for a fixed val-loss print "
                         "(default: sibling 'val' dir of --games-dirs[0])")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = M.ChuNet(a.d_model, a.blocks, a.heads).to(device)
    if a.init_from and os.path.exists(a.init_from):
        state = torch.load(a.init_from, map_location="cpu",
                           weights_only=True)
        net.load_state_dict(state["model"] if "model" in state else state)
        print(f"[train] init from {a.init_from}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)

    decay = a.lr_decay_steps or a.steps

    def lr_at(step):
        if step < a.warmup_steps:
            return a.lr * step / max(a.warmup_steps, 1)
        t = (step - a.warmup_steps) / max(decay - a.warmup_steps, 1)
        return 1e-4 + 0.5 * (a.lr - 1e-4) * (1 + np.cos(np.pi * min(t, 1)))

    buf = B.Buffer(a.games_dirs, a.max_positions)
    print(f"[train] buffer: {len(buf)} positions", flush=True)
    if len(buf) < a.batch:
        print("[train] not enough data", flush=True)
        return 1
    wdl_targets = buf.value_targets(a.value_target, a.lam)

    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(a.step0, a.step0 + a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = rng.choice(len(buf), size=a.batch, replace=len(buf) < a.batch)
        packed, prog, (pol_idx, pol_p) = buf.batch(idx)
        planes = M.unpack_batch(packed.tobytes(), prog).to(device)
        target_v = torch.from_numpy(wdl_targets[idx]).to(device)

        policy, wdl = net(planes)
        logp = F.log_softmax(policy.float(), dim=1)
        # sparse soft cross-entropy over the visited moves
        bi = torch.from_numpy(np.concatenate(
            [np.full(len(p), i) for i, p in enumerate(pol_idx)])).to(device)
        mi = torch.from_numpy(np.concatenate(pol_idx)).to(device)
        mp = torch.from_numpy(np.concatenate(pol_p)).to(device)
        loss_p = -(mp * logp[bi, mi]).sum() / a.batch
        loss_v = -(target_v * F.log_softmax(wdl.float(), dim=1)).sum() / a.batch
        loss = loss_p + a.value_weight * loss_v

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step % a.log_every == 0:
            print(f"[train] step {step} loss {loss.item():.4f} "
                  f"(p {loss_p.item():.4f} v {loss_v.item():.4f}) "
                  f"lr {lr_at(step):.2e} "
                  f"({(time.time() - t0) / (step - a.step0 + 1):.2f}s/step)",
                    flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == a.step0 + a.steps:
            tmp = a.out + ".tmp"
            torch.save({"model": net.state_dict(), "step": step + 1,
                        "config": vars(a)}, tmp)
            os.rename(tmp, a.out)  # atomic publish for the eval server

    # Held-out validation loss on a fixed position set (stationary metric,
    # unlike the noisy single-batch step prints). Auto-detects a "val" dir
    # next to the first games dir when --val-games-dir is not given.
    val_dir = a.val_games_dir
    if not val_dir:
        cand = Path(a.games_dirs[0]).parent / "val"
        if cand.is_dir():
            val_dir = str(cand)
    if val_dir and list(Path(val_dir).rglob("game_*.json")):
        vbuf = B.Buffer([val_dir])
        vrng = np.random.default_rng(0)  # fixed subsample every run
        vidx = vrng.choice(len(vbuf), size=min(4096, len(vbuf)),
                           replace=False)
        vpacked, vprog, (vi, vp) = vbuf.batch(vidx)
        vt = torch.from_numpy(
            vbuf.value_targets(a.value_target, a.lam)[vidx]).to(device)
        vplanes = M.unpack_batch(vpacked.tobytes(), vprog).to(device)
        vbi = torch.from_numpy(np.concatenate(
            [np.full(len(p), i) for i, p in enumerate(vi)])).to(device)
        vmi = torch.from_numpy(np.concatenate(vi)).to(device)
        vmp = torch.from_numpy(np.concatenate(vp)).to(device)
        lp = lv = 0.0
        with torch.no_grad():
            for s in range(0, len(vidx), 512):
                r = slice(s, min(s + 512, len(vidx)))
                policy, wdl = net(vplanes[r])
                logp = F.log_softmax(policy.float(), dim=1)
                sel = (vbi >= s) & (vbi < s + 512)
                lp += -(vmp[sel] * logp[vbi[sel] - s, vmi[sel]]).sum().item()
                lv += -(vt[r] *
                        F.log_softmax(wdl.float(), dim=1)).sum().item()
        lp /= len(vidx)
        lv /= len(vidx)
        print(f"[train] val loss {lp + lv:.4f} (p {lp:.4f} v {lv:.4f}) "
              f"on {len(vidx)} held-out positions", flush=True)
    print("[train] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
