#!/usr/bin/env python3
"""Build one AcceptanceMap per recoil-scan condition (cluster).

  python3 scripts/build_acceptance_maps.py RUNS_DIR/rscan_v1 CONDITIONS.json OUT_DIR [COND]

Reads every job of a condition (<cond>/jNN/out.root). Features are theta_x,
theta_y, delta = |p|/p_compound - 1 (p_compound from the beam KE at the
vertex) and vertex z; the label is dsssdRecoil. Holds out 10% of events to
report the binned calibration and log-loss, then refits on everything and
writes OUT_DIR/<cond>.pkl plus summary.json.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cluster"))
from nnemu.recoil import AcceptanceMap  # noqa: E402
from validation_reactions import REACTIONS  # noqa: E402
sys.path.insert(0, os.path.expanduser("~/codes/Ancalagon/reactions"))

AMU = 931.49432
ME = {"39K": -33.8072, "38K": -28.80075, "19F": -1.48744, "22Ne": -8.02472, "15O": 2.85561}
A_OF = {"39K": 39, "38K": 38, "19F": 19, "22Ne": 22, "15O": 15}


def load(cond_dir, beam):
    mb = A_OF[beam] * AMU + ME[beam]
    F, Y = [], []
    for f in sorted(glob.glob(os.path.join(cond_dir, "j*", "out.root"))):
        with uproot.open(f) as r:
            t = r["Truth"].arrays(["recoilPx", "recoilPy", "recoilPz", "beamKeMeV", "vertexZ_cm"], library="np")
            y = r["BgoEvent"]["dsssdRecoil"].array(library="np")
        pc = np.sqrt((t["beamKeMeV"] + mb) ** 2 - mb**2)
        p = np.sqrt(t["recoilPx"] ** 2 + t["recoilPy"] ** 2 + t["recoilPz"] ** 2)
        F.append(np.column_stack([t["recoilPx"] / t["recoilPz"], t["recoilPy"] / t["recoilPz"], p / pc - 1,
                                  t["vertexZ_cm"]]).astype(np.float32))
        Y.append(y)
    return np.vstack(F), np.concatenate(Y)


def main() -> int:
    scan_dir, conds_path, out = sys.argv[1:4]
    only = sys.argv[4] if len(sys.argv) > 4 else None
    os.makedirs(out, exist_ok=True)
    summary = {}
    for c in json.load(open(conds_path)):
        if only and c["cond"] != only:
            continue
        dst = os.path.join(out, c["cond"] + ".pkl")
        if os.path.exists(dst):
            continue
        F, y = load(os.path.join(scan_dir, c["cond"]), REACTIONS[c["reaction"]][0])
        rng = np.random.default_rng(0)
        te = rng.random(len(y)) < 0.1
        m = AcceptanceMap(c["theta_max"], c["delta_max"]).fit(F[~te], y[~te])
        p = m.predict(F[te])
        ll = float(-np.mean(y[te] * np.log(np.clip(p, 1e-6, 1)) + (1 - y[te]) * np.log(np.clip(1 - p, 1e-6, 1))))
        # calibration: predicted vs observed in bins of predicted probability
        bins = np.linspace(0, 1, 11)
        k = np.clip(np.digitize(p, bins) - 1, 0, 9)
        cal = [(float(p[k == i].mean()), float(y[te][k == i].mean()), int((k == i).sum())) for i in range(10) if (k == i).any()]
        m = AcceptanceMap(c["theta_max"], c["delta_max"]).fit(F, y)
        with open(dst, "wb") as fo:
            pickle.dump(m, fo)
        summary[c["cond"]] = dict(n=int(len(y)), T=float(y.mean()), logloss=ll, calibration=cal)
        print(f"{c['cond']:22s} n={len(y)} T={y.mean():.3f} logloss={ll:.4f} "
              f"max|cal| {max(abs(a - b) for a, b, nn in cal if nn > 200):.3f}", flush=True)
    with open(os.path.join(out, f"summary_{only or 'all'}.json"), "w") as fo:
        json.dump(summary, fo, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
