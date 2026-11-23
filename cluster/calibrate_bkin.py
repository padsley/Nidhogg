#!/usr/bin/env python3
"""Measure, per validation reaction, the beam energy loss per cm of target
gas (at 4 Torr, 300 K), so make_validation.py can choose BKIN for a wanted
mean vertex z at any pressure:

    BKIN(P, z) = E_res_lab + slope_4Torr * (P / 4) * (z + 5.315)

Method: run Ancalagon locally (build-nn) with BKIN = E_res_lab + delta for a
small delta, read the mean vertex depth from Truth, slope = delta / depth.
Then check one target point. Energy loss is close to linear over the few-mm
to few-cm depths used. Writes bkin_slopes.json.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import uproot

sys.path.insert(0, os.path.expanduser("~/codes/Ancalagon/reactions"))
from reaction_writer import ReactionSpec  # noqa: E402
from validation_reactions import REACTIONS  # noqa: E402

BIN = os.path.expanduser("~/codes/Ancalagon/build-nn/Ancalagon")
HALF = 5.315


def spec_for(name, bkin):
    b, t, r, q, er, lv, br = REACTIONS[name]
    return ReactionSpec.from_dict({"beam": b, "target": t, "recoil": r, "charge_state": q,
                                   "eres_mev": er, "branches": [[-1, 100, 0]],
                                   "bkin_mev": bkin, "rwid_mev": 1e-4})


def mean_depth(name, bkin, pressure=4.0, n=60):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        spec_for(name, bkin).write(os.path.join(d, "r.reaction"))
        env = dict(os.environ, GAS_PRESSURE_TORR=str(pressure), GAS_TEMPERATURE_K="300",
                   OUTPUT_MODE="compact", REACTION_INPUT=os.path.join(d, "r.reaction"),
                   TRAJ_QUIET="1", RANDOM_SEED="7")
        p = subprocess.run([BIN, "--track-reaction", str(n)], cwd=d, env=env,
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"{name} bkin={bkin}: {p.stderr[-400:]}")
        z = uproot.open(os.path.join(d, "dragon_hits.root"))["Truth"]["vertexZ_cm"].array(library="np")
    return float(z.mean() + HALF)


def main() -> None:
    from concurrent.futures import ThreadPoolExecutor
    out = {}

    def one(name):
        e0 = spec_for(name, None).beam_ke_resonance_mev
        delta = 0.02
        d1 = mean_depth(name, e0 + delta)
        slope = delta / d1
        # check: aim for depth 5 cm (z ~ -0.3)
        bk = e0 + slope * 5.0
        d2 = mean_depth(name, bk)
        slope2 = (bk - e0) / d2
        return name, dict(e_res_lab=e0, slope_4torr=slope2, first_slope=slope, check_depth=d2)

    with ThreadPoolExecutor(6) as ex:
        for name, r in ex.map(one, REACTIONS):
            out[name] = r
            print(f"{name:7s} E_res_lab={r['e_res_lab']:.4f}  slope(4 Torr)={r['slope_4torr']:.5f} MeV/cm "
                  f"(first {r['first_slope']:.5f}; aimed depth 5.00 -> got {r['check_depth']:.2f} cm)")
    with open("bkin_slopes.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
