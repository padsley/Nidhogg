#!/usr/bin/env python3
"""Closure-test pseudo-data from full-cascade Ancalagon runs (cluster).

  python3 scripts/make_pseudodata.py RUNS_DIR OUT_DIR RUN_NAME [--n 300 1000 3000 10000]
        [--max-reps 100] [--response responses/s1805.yaml]

The run's events are shuffled once; for each sample size n (GENERATED
reactions) it takes up to max-reps DISJOINT blocks of n events. Each block is
one pseudo-experiment, histogrammed coherently (every head comes from the
same events, so correlations between heads are realistic) for DSSSD-gated
events with the response applied once per event:
  e_max, e_sum (E_EDGES), mult (N_MULT), crystal_max (30),
  per_crystal (30 x C_EDGES, of response-applied deposits: what data give).
Writes OUT_DIR/<run>.npz with keys n<size>_<head> of shape (reps, ...), uint16.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nnemu.ancalagon_io import read_compact  # noqa: E402
from nnemu.observables import observables  # noqa: E402
from nnemu.response import Response  # noqa: E402
from nnemu.trainset import C_EDGES, E_EDGES, N_CRYSTALS, N_MULT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("out_dir")
    ap.add_argument("run")
    ap.add_argument("--n", type=int, nargs="+", default=[300, 1000, 3000, 10000])
    ap.add_argument("--max-reps", type=int, default=100)
    ap.add_argument("--response", default=os.path.join(os.path.dirname(__file__), "..", "responses", "s1805.yaml"))
    a = ap.parse_args()
    out = os.path.join(a.out_dir, a.run + ".npz")
    if os.path.exists(out):
        print("exists", out)
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    run = read_compact(os.path.join(a.runs_dir, a.run, "out.root"))
    raw = run.dense_deposits()
    N = len(raw)
    gated = run.dsssd_recoil[:N].astype(bool)
    rng = np.random.default_rng(12345)
    dr = Response.load(a.response).apply(raw, rng)
    o = observables(dr)
    nE, nC = len(E_EDGES) - 1, len(C_EDGES) - 1
    # per-event bin indices (-1 = not filled)
    hit = o["hit"] & gated
    b_emax = np.where(hit, np.clip(np.digitize(o["e_max"], E_EDGES) - 1, -1, nE), -1)
    b_esum = np.where(hit, np.clip(np.digitize(o["e_sum"], E_EDGES) - 1, -1, nE), -1)
    b_emax[b_emax >= nE] = -1
    b_esum[b_esum >= nE] = -1
    b_mult = np.where(gated, np.minimum(o["mult"], N_MULT - 1), -1)
    b_cmax = np.where(hit, o["crystal_max"] - 1, -1)
    ci = np.clip(np.digitize(dr, C_EDGES) - 1, 0, nC)
    pc_ok = (dr > 0) & gated[:, None] & (ci < nC)
    perm = rng.permutation(N)
    arrays = dict(n_total=N, response=Response.load(a.response).name)

    def hist(idx, size):
        idx = idx[idx >= 0]
        return np.bincount(idx, minlength=size)[:size]

    for n in a.n:
        reps = min(a.max_reps, N // n)
        H = {k: [] for k in ["e_max", "e_sum", "mult", "crystal_max", "per_crystal"]}
        for r in range(reps):
            ev = perm[r * n:(r + 1) * n]
            H["e_max"].append(hist(b_emax[ev], nE))
            H["e_sum"].append(hist(b_esum[ev], nE))
            H["mult"].append(hist(b_mult[ev], N_MULT))
            H["crystal_max"].append(hist(b_cmax[ev], N_CRYSTALS))
            cc = np.nonzero(pc_ok[ev])
            H["per_crystal"].append(np.bincount(cc[1] * nC + ci[ev][cc], minlength=N_CRYSTALS * nC)
                                    .reshape(N_CRYSTALS, nC))
        for k, v in H.items():
            arrays[f"n{n}_{k}"] = np.array(v, dtype=np.uint16)
        print(f"n={n}: {reps} replicas, mean gated hits {np.mean([h.sum() for h in H['e_max']]):.0f}", flush=True)
    np.savez_compressed(out + ".partial.npz", **arrays)
    os.rename(out + ".partial.npz", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
