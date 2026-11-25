#!/usr/bin/env python3
"""Diagnose the per-crystal head's error against full Ancalagon runs.

  python3 scripts/diag_per_crystal.py VALHIST_DIR PMAPS_DIR MODEL.pt [...] [--every 3] [--sel all]
        [--out diag.json]

Uses the same mixture prediction as eval_val.py (ensemble mean) and reports:
  1. deviance/ndf vs. truth statistics, by binomial thinning of the truth
     counts (fractions 0.03..1). Pure statistical agreement stays ~1; a
     systematic shape error grows roughly linearly with N.
  2. where the excess deviance sits: by distance of the bin from the nearest
     photopeak / Compton edge of the run's gamma energies, by energy, by
     crystal; with the mean signed pull (sign of the bias).
  3. chi2/ndf after smearing truth and prediction with the response's
     resolution (Gaussian, per-crystal a,b,c), i.e. what a fit sees.
     Variance of the smeared truth = sum_j R_ij^2 n_j, so E[chi2/ndf] = 1
     under no model error even though smeared bins are correlated.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_val import load_model, predict_mixture  # noqa: E402
from nnemu.recoil import ACCEPTANCE_PARAMS, OPEN_ACCEPTANCE  # noqa: E402
from nnemu.response import Response  # noqa: E402
from nnemu.trainset import C_EDGES  # noqa: E402

ME = 0.51099895
CC = 0.5 * (C_EDGES[1:] + C_EDGES[:-1])
FRACS = [0.03, 0.1, 0.3, 1.0]


def dev_terms(n, mu):
    return 2 * (mu - n + np.where(n > 0, n * np.log(np.maximum(n, 1e-300) / mu), 0.0))


def smear_matrix(resp, crystal):
    e = np.zeros((len(CC), 30))
    e[:, crystal] = CC
    sig = np.maximum(resp.sigma(e)[:, crystal], 1e-4)  # sigma(E) at source bin centre
    from scipy.stats import norm
    cdf = norm.cdf((C_EDGES[:, None] - CC[None, :]) / sig[None, :])
    return np.diff(cdf, axis=0)  # R[i, j]: source bin j -> observed bin i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("valhist")
    ap.add_argument("pmaps")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--sel", default="all", choices=["all", "gated"])
    ap.add_argument("--response", default=os.path.join(os.path.dirname(__file__), "..", "responses", "s1805.yaml"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(16)
    models = [load_model(m)[0] for m in a.models]
    resp = Response.load(a.response)
    R = [smear_matrix(resp, c) for c in range(30)]
    rng = np.random.default_rng(0)

    thin = {f: [0.0, 0] for f in FRACS}
    dist_edges = np.array([-np.inf, -0.6, -0.25, -0.05, 0.05, 0.25, 0.6, np.inf])  # MeV from nearest peak
    cats = ["peak(+-50keV)", "near peak", "compton-edge(+-100keV)", "continuum", "below 0.3MeV", "above max peak"]
    by_cat = {c: [0.0, 0, 0.0] for c in cats}  # dev, nbins, sum signed pull
    by_E = np.zeros((len(CC), 3))
    by_cr = np.zeros((30, 3))
    smeared = [0.0, 0]
    per_run = []
    files = sorted(glob.glob(os.path.join(a.valhist, "*.npz")))[::a.every]
    for f in files:
        name = os.path.basename(f)[:-4]
        d = np.load(f)
        parts = name.split("_")
        pm = json.load(open(os.path.join(a.pmaps, f"{parts[0]}_{parts[2]}_{parts[3]}.json")))
        acc = OPEN_ACCEPTANCE if a.sel == "all" else np.array([pm["params"][k] for k in ACCEPTANCE_PARAMS])
        zc = 0.5 * (d["z_edges"][1:] + d["z_edges"][:-1])
        zh = d["z_hist"].astype(float)
        keep = zh > 0
        paths = [p[p > 0] for p in d["path_gammas"]]
        N = float(d["n_events"])
        preds = [predict_mixture(m, paths, d["path_prob"], zc[keep], zh[keep] / zh.sum(), float(d["beta"]),
                                 float(d["p0"]), acc) for m in models]
        mu = N * np.mean([p["per_crystal"] for p in preds], 0)  # (30, nbins)
        n = d[f"{a.sel}_per_crystal"].astype(float)
        # 1. thinning
        for fr in FRACS:
            nt = n if fr == 1.0 else rng.binomial(n.astype(np.int64), fr).astype(float)
            ok = fr * mu >= 5
            thin[fr][0] += dev_terms(nt[ok], fr * mu[ok]).sum()
            thin[fr][1] += ok.sum()
        # 2. location of excess
        egs = np.unique(np.concatenate(paths))
        ce = egs - egs / (1 + 2 * egs / ME)  # Compton edge energy deposit
        ok = mu >= 5
        dt = dev_terms(n, mu)
        pull = (n - mu) / np.sqrt(np.maximum(mu, 1e-9))
        dpk = CC[:, None] - egs[None, :]
        nearest = dpk[np.arange(len(CC)), np.abs(dpk).argmin(1)]
        near_ce = np.abs(CC[:, None] - ce[None, :]).min(1) <= 0.1
        cat = np.where(CC < 0.3, 4, np.where(CC > egs.max() + 0.05, 5, np.where(np.abs(nearest) <= 0.05, 0, np.where(
            np.abs(nearest) <= 0.25, 1, np.where(near_ce, 2, 3)))))
        for k, c in enumerate(cats):
            m = ok & (cat == k)[None, :]
            by_cat[c][0] += dt[m].sum()
            by_cat[c][1] += m.sum()
            by_cat[c][2] += pull[m].sum()
        for arr, ax in [(by_E, 0), (by_cr, 1)]:
            arr[:, 0] += np.where(ok, dt, 0).sum(ax)
            arr[:, 1] += ok.sum(ax)
            arr[:, 2] += np.where(ok, pull, 0).sum(ax)
        # 3. smeared
        sm_c, sm_n = 0.0, 0
        for c in range(30):
            s, sm, var = R[c] @ n[c], R[c] @ mu[c], (R[c] ** 2) @ n[c]
            okc = sm >= 20
            sm_c += ((s[okc] - sm[okc]) ** 2 / np.maximum(var[okc], 1)).sum()
            sm_n += okc.sum()
        smeared[0] += sm_c
        smeared[1] += sm_n
        raw_r = dt[ok].sum() / max(ok.sum(), 1)
        per_run.append(dict(run=name, N=N, raw=raw_r, smeared=sm_c / max(sm_n, 1)))
        print(f"{name:28s} N={N:.0f} raw {raw_r:.2f} smeared {sm_c / max(sm_n, 1):.2f}", flush=True)

    print(f"\n=== {a.sel}, {len(files)} runs ===")
    print("1. deviance/ndf vs truth-statistics fraction (noise-only -> ~1; systematic grows ~ N):")
    for fr in FRACS:
        print(f"   {fr:5.2f}: {thin[fr][0] / thin[fr][1]:.3f}  (ndf {thin[fr][1]})")
    print("2. by bin category: dev/ndf, share of total excess (dev-ndf), mean signed pull")
    tot_ex = sum(v[0] - v[1] for v in by_cat.values())
    for c, (dv, nb, ps) in by_cat.items():
        if nb:
            print(f"   {c:24s} {dv / nb:6.2f}  excess share {(dv - nb) / tot_ex:6.1%}  pull {ps / nb:+.3f}  nbins {nb}")
    print("   by energy (MeV): dev/ndf, mean pull")
    for lo in range(0, 15):
        m = (CC >= lo) & (CC < lo + 1)
        dv, nb, ps = by_E[m].sum(0)
        if nb:
            print(f"   {lo:2d}-{lo + 1:2d}: {dv / nb:6.2f} {ps / nb:+.3f}  ({nb:.0f})")
    print("   by crystal: dev/ndf")
    print("   " + " ".join(f"{c}:{by_cr[c, 0] / max(by_cr[c, 1], 1):.2f}" for c in range(30)))
    print(f"3. smeared ({a.response}) chi2/ndf: {smeared[0] / smeared[1]:.3f}  (ndf {smeared[1]})")
    sm = np.array([r["smeared"] for r in per_run])
    print(f"   per-run smeared median {np.median(sm):.2f}, 90% {np.percentile(sm, 90):.2f}")
    if a.out:
        json.dump(dict(thin={str(k): [float(x) for x in v] for k, v in thin.items()}, by_cat={k: [float(x) for x in v] for k, v in by_cat.items()}, by_E=by_E.tolist(),
                       by_crystal=by_cr.tolist(), smeared=[float(x) for x in smeared], per_run=per_run), open(a.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
