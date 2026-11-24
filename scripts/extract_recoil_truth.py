#!/usr/bin/env python3
"""Extract recoil-acceptance training data from full-cascade validation runs
(cluster). For every run in a batch manifest, sample up to --per-run events
and write one npz with:
  X       (N, 9) features: nnemu.recoil.FEATURES
  y       (N,)   dsssdRecoil
  run_id  (N,)   index into `runs`
  runs, reactions
Recoil momentum and p0 come from Truth: p0 = |p_recoil + sum p_gamma,lab|.

  python3 scripts/extract_recoil_truth.py RUNS_DIR MANIFEST OUT.npz [--per-run 20000]
"""
import argparse
import json
import os
import sys

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cluster"))
from nnemu.recoil import condition_columns, recoil_features  # noqa: E402
from validation_reactions import REACTIONS  # noqa: E402

NUC = {"1H": (1, 1), "4He": (2, 4), "40Ca": (20, 40), "39Ca": (20, 39), "20Ne": (10, 20),
       "23Na": (11, 23), "26Mg": (12, 26), "19Ne": (10, 19), "39K": (19, 39), "38K": (19, 38),
       "19F": (9, 19), "22Ne": (10, 22), "15O": (8, 15)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--per-run", type=int, default=20000)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    rng = np.random.default_rng(0)
    Xs, ys, ids, names, reacs = [], [], [], [], []
    for i, m in enumerate(man):
        beam, targ, rec = REACTIONS[m["reaction"]][:3]
        Zr, Ar = NUC[rec]
        _, Ab = NUC[beam]
        with uproot.open(os.path.join(a.runs_dir, m["tag"], "out.root")) as f:
            n = int(f["Truth"].num_entries)
            sel = np.sort(rng.choice(n, min(n, a.per_run), replace=False))
            t = f["Truth"].arrays(["vertexZ_cm", "beamKeMeV", "recoilPx", "recoilPy", "recoilPz",
                                   "gammaELab", "gammaCosThetaLab", "gammaPhiLab"], library="ak")[sel]
            y = f["BgoEvent"]["dsssdRecoil"].array(library="np")[sel]
        el, cl, pl = t["gammaELab"], t["gammaCosThetaLab"], t["gammaPhiLab"]
        sl = np.sqrt(np.maximum(1 - cl**2, 0.0))
        gx = ak.to_numpy(ak.sum(el * sl * np.cos(pl), axis=1))
        gy = ak.to_numpy(ak.sum(el * sl * np.sin(pl), axis=1))
        gz = ak.to_numpy(ak.sum(el * cl, axis=1))
        rx, ry, rz = (ak.to_numpy(t[k]) for k in ("recoilPx", "recoilPy", "recoilPz"))
        p0 = np.sqrt((rx + gx) ** 2 + (ry + gy) ** 2 + (rz + gz) ** 2)
        feat = recoil_features(rx, ry, rz, p0, ak.to_numpy(t["vertexZ_cm"]).astype(float))
        e_per_u = float(np.mean(ak.to_numpy(t["beamKeMeV"]))) / Ab
        cond = condition_columns(len(sel), m["pressure"], Ar, Zr, e_per_u, targ == "4He")
        Xs.append(np.hstack([feat, cond]).astype(np.float32))
        ys.append(y.astype(np.int8))
        ids.append(np.full(len(sel), i, np.int32))
        names.append(m["tag"])
        reacs.append(m["reaction"])
        if i % 50 == 0:
            print(i, m["tag"], f"recoil frac {y.mean():.3f}", flush=True)
    np.savez_compressed(a.out, X=np.vstack(Xs), y=np.concatenate(ys), run_id=np.concatenate(ids),
                        runs=np.array(names), reactions=np.array(reacs))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
