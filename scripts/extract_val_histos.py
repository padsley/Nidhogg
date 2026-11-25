#!/usr/bin/env python3
"""Per-run truth histograms of full-cascade validation runs, in the
emulator's own head definitions (cluster).

  python3 scripts/extract_val_histos.py RUNS_DIR MANIFEST OUT_DIR RUN_NAME [RUN_NAME ...]
        [--max-events 500000] [--response responses/s1805.yaml]

For each run, writes OUT_DIR/<run>.npz:
  n_events; e_max, e_sum, mult, crystal_max, per_crystal (counts, s1805
  response with nominal nuisances; per_crystal raw), for all events
  ('all_') and DSSSD-gated events ('gated_'); vertex-z histogram (z_edges
  -8..8 cm, 0.25 cm), mean beta and p0; cascade paths (energies,
  probabilities) from the manifest.
"""
import argparse
import json
import os
import sys

import awkward as ak
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nnemu.ancalagon_io import read_compact  # noqa: E402
from nnemu.observables import observables  # noqa: E402
from nnemu.response import Response  # noqa: E402
from nnemu.trainset import C_EDGES, E_EDGES, N_CRYSTALS, N_MULT  # noqa: E402

Z_EDGES = np.arange(-8.0, 8.0 + 1e-9, 0.25)


def heads(dep_resp, dep_raw):
    o = observables(dep_resp)
    h = o["hit"]
    pc = np.zeros((N_CRYSTALS, len(C_EDGES) - 1))
    for c in range(N_CRYSTALS):
        v = dep_raw[:, c]
        pc[c] = np.histogram(v[v > 0], C_EDGES)[0]
    return dict(e_max=np.histogram(o["e_max"][h], E_EDGES)[0], e_sum=np.histogram(o["e_sum"][h], E_EDGES)[0],
                mult=np.bincount(np.minimum(o["mult"], N_MULT - 1), minlength=N_MULT),
                crystal_max=np.bincount(o["crystal_max"], minlength=N_CRYSTALS + 1)[1:], per_crystal=pc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("manifest")
    ap.add_argument("out_dir")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--max-events", type=int, default=500_000)
    ap.add_argument("--response", default=os.path.join(os.path.dirname(__file__), "..", "responses", "s1805.yaml"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    man = {m["tag"]: m for m in json.load(open(a.manifest))}
    resp = Response.load(a.response)
    rng = np.random.default_rng(0)
    for name in a.runs:
        out = os.path.join(a.out_dir, name + ".npz")
        if os.path.exists(out):
            continue
        run = read_compact(os.path.join(a.runs_dir, name, "out.root"))
        n = min(run.n_events, a.max_events)
        raw = run.dense_deposits()[:n]
        tr = run.truth
        z = ak.to_numpy(tr["vertexZ_cm"][:n]).astype(float)
        beta = ak.to_numpy(tr["emitterBeta"][:n]).astype(float)
        el, cl, pl = tr["gammaELab"][:n], tr["gammaCosThetaLab"][:n], tr["gammaPhiLab"][:n]
        sl = np.sqrt(np.maximum(1 - cl**2, 0.0))
        p0 = np.sqrt((ak.to_numpy(tr["recoilPx"][:n]) + ak.to_numpy(ak.sum(el * sl * np.cos(pl), axis=1))) ** 2
                     + (ak.to_numpy(tr["recoilPy"][:n]) + ak.to_numpy(ak.sum(el * sl * np.sin(pl), axis=1))) ** 2
                     + (ak.to_numpy(tr["recoilPz"][:n]) + ak.to_numpy(ak.sum(el * cl, axis=1))) ** 2)
        gated = run.dsssd_recoil[:n].astype(bool)
        dr = resp.apply(raw, rng)
        arrays = dict(n_events=n, z_hist=np.histogram(z, Z_EDGES)[0], z_edges=Z_EDGES,
                      beta=beta.mean(), p0=p0.mean(), recoil_frac=gated.mean())
        for sel, m in [("all", slice(None)), ("gated", gated)]:
            for k, v in heads(dr[m], raw[m]).items():
                arrays[f"{sel}_{k}"] = v
        paths = man[name]["paths"]
        mx = max(len(p[1]) for p in paths)
        arrays["path_prob"] = np.array([p[0] for p in paths])
        arrays["path_gammas"] = np.array([sorted(p[1], reverse=True) + [0.0] * (mx - len(p[1])) for p in paths])
        np.savez(out, **arrays)
        print(name, n, f"recoil {gated.mean():.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
