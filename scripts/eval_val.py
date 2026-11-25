#!/usr/bin/env python3
"""End-to-end test of a trained emulator against full Ancalagon validation
runs (truth histograms from scripts/extract_val_histos.py).

  python3 scripts/eval_val.py VALHIST_DIR PMAPS_DIR MODEL.pt [MODEL.pt ...] [--out results.json]

With several models, the prediction is the ensemble mean of probabilities.

For each run, the prediction (per generated event) is
    sum_paths P(path) * sum_z w(z) * S_NN(path, z, beta, p0, acceptance)
using the run's branching paths, its true vertex-z histogram (0.25 cm
nodes), its mean beta and p0, an OPEN window for the ungated heads, and its
condition's fitted 14-parameter window (PMAPS_DIR/<cond>.json) for the gated
heads. Nominal response nuisances (scale 1, shift 0).

Per head it reports the Poisson deviance/ndf of the truth counts against
N*prediction, over bins with expected counts >= 5. Only truth statistics
enter, so this is the emulator's total error: network + factorization +
acceptance parametrisation.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nnemu.model import Emulator, encode_context  # noqa: E402
from nnemu.recoil import ACCEPTANCE_PARAMS, OPEN_ACCEPTANCE  # noqa: E402

HEADS = ["e_max", "e_sum", "mult", "crystal_max", "per_crystal"]


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Emulator(doppler=ck.get("args", {}).get("doppler", False))
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


@torch.no_grad()
def predict_mixture(model, paths, probs, z_nodes, z_w, beta, p0, acc):
    """Mixture prediction; returns dict head -> probabilities per generated
    event (without the 'other' entry)."""
    W = np.array([pp * w for pp in probs for w in z_w])
    Z = np.tile(z_nodes, len(paths))
    n = len(W)
    gam = np.zeros((n, 6), np.float32)
    row = 0
    for p in paths:
        e = np.sort(np.asarray(p, np.float32))[::-1]
        gam[row:row + len(z_nodes), :len(e)] = e
        row += len(z_nodes)
    ctx = encode_context(Z, np.full(n, beta), np.full(n, p0), np.tile(acc, (n, 1)),
                         np.ones(n), np.zeros(n))
    out = {h: 0.0 for h in HEADS}
    for i in range(0, n, 128):
        o = model(torch.from_numpy(gam[i:i + 128]), torch.from_numpy(ctx[i:i + 128]))
        w = torch.from_numpy(W[i:i + 128]).float()
        for h in HEADS:
            p = torch.exp(o[h])
            ww = w.view(-1, *([1] * (p.dim() - 1)))
            out[h] = out[h] + (p * ww).sum(0).numpy()
    return {h: (v[..., :-1]) for h, v in out.items()}


def deviance(n, mu, min_mu=5.0):
    n, mu = np.ravel(n).astype(float), np.ravel(mu)
    ok = mu >= min_mu
    t = np.where(n[ok] > 0, n[ok] * np.log(n[ok] / mu[ok]), 0.0)
    return float(2 * np.sum(mu[ok] - n[ok] + t)), int(ok.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("valhist")
    ap.add_argument("pmaps")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(16)
    models = []
    for mp in a.models:
        m, ck = load_model(mp)
        models.append(m)
        print(f"{mp}: epoch {ck['epoch']}, response {ck['meta']['response']}")
    res = []
    for f in sorted(glob.glob(os.path.join(a.valhist, "*.npz"))):
        name = os.path.basename(f)[:-4]
        d = np.load(f)
        parts = name.split("_")
        cond = f"{parts[0]}_{parts[2]}_{parts[3]}"
        pm = json.load(open(os.path.join(a.pmaps, cond + ".json")))
        acc_g = np.array([pm["params"][k] for k in ACCEPTANCE_PARAMS])
        zc = 0.5 * (d["z_edges"][1:] + d["z_edges"][:-1])
        zh = d["z_hist"].astype(float)
        keep = zh > 0
        zn, zw = zc[keep], zh[keep] / zh.sum()
        paths = [p[p > 0] for p in d["path_gammas"]]
        N = float(d["n_events"])
        r = dict(run=name, n=N, recoil_frac=float(d["recoil_frac"]))
        for sel, acc in [("all", OPEN_ACCEPTANCE), ("gated", acc_g)]:
            preds = [predict_mixture(m, paths, d["path_prob"], zn, zw, float(d["beta"]), float(d["p0"]), acc)
                     for m in models]
            pred = {h: np.mean([q[h] for q in preds], 0) for h in HEADS}
            for h in HEADS:
                dev, ndf = deviance(d[f"{sel}_{h}"], N * pred[h])
                r[f"{sel}/{h}"] = (dev, ndf)
            r[f"{sel}/p_hit"] = (float(pred["e_max"].sum()), float(d[f"{sel}_e_max"].sum() / N))
        res.append(r)
        print(f"{name:28s} " + " ".join(f"{h}={r['gated/' + h][0] / max(r['gated/' + h][1], 1):.2f}"
                                        for h in HEADS) + f"  P(gated hit) nn {r['gated/p_hit'][0]:.4f} "
              f"true {r['gated/p_hit'][1]:.4f}", flush=True)
    for sel in ["all", "gated"]:
        print(f"\n[{sel}] median / 90% deviance per ndf over {len(res)} runs:")
        for h in HEADS:
            v = np.array([r[f"{sel}/{h}"][0] / max(r[f"{sel}/{h}"][1], 1) for r in res])
            print(f"  {h:12s} {np.median(v):6.2f} {np.percentile(v, 90):6.2f}")
        dp = np.array([r[f"{sel}/p_hit"][0] - r[f"{sel}/p_hit"][1] for r in res])
        print(f"  P(hit) nn-true: mean {dp.mean():+.4f} sd {dp.std():.4f} max|.| {np.abs(dp).max():.4f}")
    if a.out:
        json.dump(res, open(a.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
