#!/usr/bin/env python3
"""Train the bin-by-bin emulator on training-set shards (CPU).

  python3 scripts/train.py SHARD_DIR OUT_DIR [--epochs 12] [--batch 64]
        [--crystals-per-step 6] [--seed 0] [--val-frac 0.05] [--threads 16]

Loss: multinomial cross-entropy per generated event, summed over heads
(nnemu.model.loss_fn). Validation reports KL(target || model) per head
(cross-entropy minus target entropy), which is 0 for a perfect emulator of
the targets' expectation. Targets carry Poisson/library noise, so the
reachable floor is above 0. Saves OUT_DIR/model.pt (best validation) and
history.json.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nnemu.model import Emulator, encode_context, loss_fn, with_other  # noqa: E402
from nnemu.trainset import N_CRYSTALS  # noqa: E402

HEADS = ["e_max", "e_sum", "mult", "crystal_max", "per_crystal"]


def load_shards(files):
    g, ctx, t = [], [], {h: [] for h in HEADS}
    meta = None
    for f in files:
        d = np.load(f)
        g.append(d["in_gammas"])
        ctx.append(encode_context(d["in_z"], d["in_beta"], d["in_p0"], d["in_acceptance"],
                                  d["in_res_scale"], d["in_thr_shift"]))
        p_pass = d["t_p_pass"][:, 0]
        t["e_max"].append(with_other(d["t_e_max"]).astype(np.float32))
        t["e_sum"].append(with_other(d["t_e_sum"]).astype(np.float32))
        t["crystal_max"].append(with_other(d["t_crystal_max"]).astype(np.float32))
        t["mult"].append(np.concatenate([d["t_mult"], (1 - p_pass)[:, None]], 1).astype(np.float32))
        t["per_crystal"].append(with_other(d["t_per_crystal"].astype(np.float32)).astype(np.float16))
        meta = dict(response=str(d["response"]), response_hash=str(d["response_hash"]), n_events=int(d["n_events"]))
    return (np.concatenate(g).astype(np.float32), np.concatenate(ctx),
            {h: np.concatenate(v) for h, v in t.items()}, meta)


def load_cached(shard_dir, val_frac, max_shards=None):
    """Consolidate shards once into SHARD_DIR/cache/{train,val}_*.npy and
    memory-map them. Several training processes (ensemble members) then
    share one copy through the OS page cache instead of each holding the
    full set in RAM. Validation = the fixed last shards (shared by all
    members)."""
    cache = os.path.join(shard_dir, "cache")
    done = os.path.join(cache, "DONE")
    if not os.path.exists(done):
        files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))[:max_shards]
        nval = max(1, int(round(val_frac * len(files))))
        os.makedirs(cache, exist_ok=True)
        for split, fl in [("train", files[:-nval]), ("val", files[-nval:])]:
            g, ctx, t, meta = load_shards(fl)
            np.save(os.path.join(cache, f"{split}_gammas.npy"), g)
            np.save(os.path.join(cache, f"{split}_context.npy"), ctx)
            for h, v in t.items():
                np.save(os.path.join(cache, f"{split}_{h}.npy"), v)
            del g, ctx, t
        json.dump(dict(meta=meta, n_files=len(files), n_val_files=nval), open(os.path.join(cache, "meta.json"), "w"))
        open(done, "w").close()
    meta = json.load(open(os.path.join(cache, "meta.json")))["meta"]
    out = []
    for split in ("train", "val"):
        ld = lambda n: np.load(os.path.join(cache, f"{split}_{n}.npy"), mmap_mode="r")  # noqa: E731
        out.append((ld("gammas"), ld("context"), {h: ld(h) for h in HEADS}, meta))
    return out


def entropy(t):
    return -(t * np.log(np.clip(t, 1e-12, None))).sum(-1)


@torch.no_grad()
def evaluate(model, g, ctx, t, batch=256):
    model.eval()
    kl = {h: 0.0 for h in HEADS}
    n = len(g)
    for i in range(0, n, batch):
        sl = slice(i, i + batch)
        out = model(torch.from_numpy(np.ascontiguousarray(g[sl])), torch.from_numpy(np.ascontiguousarray(ctx[sl])))
        for h in HEADS:
            tt = t[h][sl].astype(np.float32)
            ce = -(tt * out[h].numpy()).sum(-1)
            k = ce - entropy(tt)
            kl[h] += float(k.sum(1).sum() if h == "per_crystal" else k.sum())
    model.train()
    return {h: v / n for h, v in kl.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--crystals-per-step", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-shards", type=int, default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    t0 = time.time()
    (g, ctx, t, meta), (gv, ctxv, tv, _) = load_cached(a.shard_dir, a.val_frac, a.max_shards)
    print(f"train {len(g)} samples, val {len(gv)} (memory-mapped cache); load {time.time() - t0:.0f}s; "
          f"response {meta['response']} ({meta['response_hash']})", flush=True)

    model = Emulator()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    steps_per_epoch = len(g) // a.batch
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    hist, best = [], np.inf
    for ep in range(a.epochs):
        perm = rng.permutation(len(g))
        t1, run = time.time(), []
        for s in range(steps_per_epoch):
            idx = np.sort(perm[s * a.batch:(s + 1) * a.batch])
            crystals = torch.from_numpy(np.sort(rng.choice(N_CRYSTALS, a.crystals_per_step, replace=False)))
            out = model(torch.from_numpy(np.ascontiguousarray(g[idx])),
                        torch.from_numpy(np.ascontiguousarray(ctx[idx])), crystals=crystals)
            tgt = {h: torch.from_numpy(t[h][idx].astype(np.float32)) for h in HEADS}
            loss, parts = loss_fn(out, tgt, crystals)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run.append(float(loss.detach()))
            if s % 500 == 0:
                print(f"ep {ep} step {s}/{steps_per_epoch} loss {np.mean(run[-500:]):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} ({time.time() - t1:.0f}s)", flush=True)
        kl = evaluate(model, gv, ctxv, tv)
        score = sum(kl.values())
        hist.append(dict(epoch=ep, train_loss=float(np.mean(run)), val_kl=kl, seconds=time.time() - t1))
        print(f"== epoch {ep}: val KL " + " ".join(f"{h}={v:.5f}" for h, v in kl.items()) + f" total={score:.5f}", flush=True)
        if score < best:
            best = score
            torch.save(dict(state=model.state_dict(), meta=meta, args=vars(a), epoch=ep, val_kl=kl),
                       os.path.join(a.out_dir, "model.pt"))
        json.dump(hist, open(os.path.join(a.out_dir, "history.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
