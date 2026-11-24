#!/usr/bin/env python3
"""Fit the 10-parameter soft-box acceptance to each recoil-scan condition
and compare it with the scan (and the GBT map) on held-out events.

  python3 scripts/fit_parametric_acceptance.py RUNS_DIR/rscan_vN CONDITIONS.json OUT_DIR [COND]
Writes OUT_DIR/<cond>.pkl (ParametricAcceptanceMap) and <cond>.json.
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cluster"))
from build_acceptance_maps import load  # noqa: E402
from nnemu.recoil import ACCEPTANCE_PARAMS, ParametricAcceptanceMap, fit_parametric_acceptance  # noqa: E402
from validation_reactions import REACTIONS  # noqa: E402


def main() -> int:
    scan_dir, conds_path, out = sys.argv[1:4]
    only = sys.argv[4] if len(sys.argv) > 4 else None
    os.makedirs(out, exist_ok=True)
    for c in json.load(open(conds_path)):
        if only and c["cond"] != only:
            continue
        F, y = load(os.path.join(scan_dir, c["cond"]), REACTIONS[c["reaction"]][0])
        rng = np.random.default_rng(0)
        te = rng.random(len(y)) < 0.1
        p, nll = fit_parametric_acceptance(F[~te], y[~te])
        m = ParametricAcceptanceMap(p)
        pt = m.predict(F[te])
        ll = float(-np.mean(y[te] * np.log(np.clip(pt, 1e-6, 1)) + (1 - y[te]) * np.log(np.clip(1 - pt, 1e-6, 1))))
        with open(os.path.join(out, c["cond"] + ".pkl"), "wb") as f:
            pickle.dump(m, f)
        res = dict(cond=c["cond"], params=dict(zip(ACCEPTANCE_PARAMS, map(float, p))), holdout_logloss=ll,
                   T_true=float(y[te].mean()), T_pred=float(pt.mean()))
        json.dump(res, open(os.path.join(out, c["cond"] + ".json"), "w"), indent=1)
        print(c["cond"], f"logloss {ll:.4f}", " ".join(f"{k}={v:.4g}" for k, v in res["params"].items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
