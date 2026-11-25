#!/usr/bin/env python3
"""Generate one training-set shard on the cluster.

  python3 scripts/make_trainset_shard.py PACKED_LIB OUT_DIR SHARD_INDEX
        [--n-samples 400] [--n-events 100000] [--response responses/s1805.yaml]

Samples are drawn with seed = SHARD_INDEX and processed in (beta, z) order
so consecutive samples reuse the same library parts. Writes
OUT_DIR/shard_<index>.npz (float16 targets, float32 inputs) atomically, and
skips existing shards.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nnemu.library import GammaLibrary  # noqa: E402
from nnemu.response import Response  # noqa: E402
from nnemu.trainset import make_targets, sample_conditions  # noqa: E402

HEADS = ["e_max", "e_sum", "mult", "crystal_max", "per_crystal", "p_pass"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packed_lib")
    ap.add_argument("out_dir")
    ap.add_argument("shard", type=int)
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--n-events", type=int, default=100000)
    ap.add_argument("--response", default=os.path.join(os.path.dirname(__file__), "..", "responses", "s1805.yaml"))
    a = ap.parse_args()
    out = os.path.join(a.out_dir, f"shard_{a.shard:05d}.npz")
    if os.path.exists(out):
        print("exists", out)
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(1_000_000 + a.shard)
    cond = sample_conditions(rng, a.n_samples)
    order = np.lexsort((cond["z"], cond["beta"]))
    lib = GammaLibrary(a.packed_lib)
    resp = Response.load(a.response)
    tgt = {h: [] for h in HEADS}
    libf = []
    t0 = time.time()
    for k, i in enumerate(order):
        r = make_targets(lib, cond, i, a.n_events, resp, rng)
        for h in HEADS:
            tgt[h].append(r[h])
        libf.append(r["lib_var_factor"])
        if k % 50 == 0:
            print(f"{k}/{a.n_samples} {time.time() - t0:.0f}s", flush=True)
    inputs = {k: np.asarray(v)[order].astype(np.float32) for k, v in cond.items()}
    arrays = {f"in_{k}": v for k, v in inputs.items()}
    arrays.update({f"t_{h}": np.stack(v).astype(np.float16 if h == "per_crystal" else np.float32)
                   for h, v in tgt.items()})
    arrays["lib_var_factor"] = np.array(libf, np.float32)
    arrays["n_events"] = np.int64(a.n_events)
    arrays["response"] = np.array(resp.name)
    arrays["response_hash"] = np.array(resp.hash)
    np.savez(out + ".partial.npz", **arrays)
    os.rename(out + ".partial.npz", out)
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
