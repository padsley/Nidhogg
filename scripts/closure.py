#!/usr/bin/env python3
"""Closure test: fit full-Ancalagon pseudo-experiments with the emulator.

  python3 scripts/closure.py RUN VALHIST_DIR PMAPS_DIR PSEUDO_DIR OUT.json MODEL.pt [MODEL.pt ...]
        [--max-reps 50] [--sizes 300 1000 3000 10000]

RUN is a '<reaction>_scheme_P<p>_z<zt>' validation run. Its level scheme
comes from cluster/validation_reactions.py; its vertex distribution, beta
and p0 from VALHIST_DIR/RUN.npz; its gated acceptance window from PMAPS_DIR.
Free parameters: branching fractions of every level with >1 branch, the
number of reactions N, and a vertex shift dz (truth: the scheme weights, the
block size n, 0). Each pseudo-experiment is fitted with nested head sets
  E1 -> +sum -> +mult -> +hitpattern -> +percrystal
and the fitted values, delta-method errors and truths are written to OUT.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "cluster"))
sys.path.insert(0, HERE)
from eval_val import load_model  # noqa: E402
from nnemu.fitter import Fitter, Scheme, Templates  # noqa: E402
from nnemu.recoil import ACCEPTANCE_PARAMS  # noqa: E402
from nnemu.response import Response  # noqa: E402
from validation_reactions import REACTIONS  # noqa: E402

HEAD_SETS = {
    "E1": ["e_max"],
    "E1+sum": ["e_max", "e_sum"],
    "+mult": ["e_max", "e_sum", "mult"],
    "+hitpat": ["e_max", "e_sum", "mult", "crystal_max"],
    "+percrystal": ["e_max", "e_sum", "mult", "crystal_max", "per_crystal"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("valhist")
    ap.add_argument("pmaps")
    ap.add_argument("pseudo")
    ap.add_argument("out")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--max-reps", type=int, default=50)
    ap.add_argument("--sizes", type=int, nargs="+", default=[300, 1000, 3000, 10000])
    ap.add_argument("--response", default=os.path.join(HERE, "..", "responses", "s1805.yaml"))
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    rx = a.run.split("_")[0]
    _, _, _, _, _, levels, branches = REACTIONS[rx]
    vh = np.load(os.path.join(a.valhist, a.run + ".npz"))
    e_entry = float(vh["path_gammas"].sum(1).max())
    sc = Scheme(levels, branches, e_entry)
    # consistency with the run's own manifest paths
    man = sorted((round(float(p), 6), tuple(np.round(g[g > 0], 4))) for p, g in zip(vh["path_prob"], vh["path_gammas"]))
    tf = sc.true_fractions()
    mine = sorted((round(float(np.prod([tf[n][i] for n, i in used])), 6), tuple(np.round(gam, 4)))
                  for gam, used in sc.paths)
    assert man == mine, f"scheme/manifest mismatch\n{man}\n{mine}"

    parts = a.run.split("_")
    pm = json.load(open(os.path.join(a.pmaps, f"{parts[0]}_{parts[2]}_{parts[3]}.json")))
    acc = np.array([pm["params"][k] for k in ACCEPTANCE_PARAMS])
    zc = 0.5 * (vh["z_edges"][1:] + vh["z_edges"][:-1])
    models = [load_model(m)[0] for m in a.models]
    tp = Templates(models, sc, zc, float(vh["beta"]), float(vh["p0"]), acc, Response.load(a.response))
    print(f"{a.run}: {len(sc.paths)} paths, {sc.n_params()} branching params; templates {time.time() - t0:.0f}s",
          flush=True)
    truth = np.concatenate([tf[n] for n in sc.free_nodes])
    ps = np.load(os.path.join(a.pseudo, a.run + ".npz"))
    fitters = {k: Fitter(sc, tp, vh["z_hist"], heads) for k, heads in HEAD_SETS.items()}
    res = dict(run=a.run, free_nodes=[int(n) for n in sc.free_nodes],
               labels=[f"b{n}->{sc.out[n][i][0]}" for n in sc.free_nodes for i in range(len(sc.out[n]))]
               + ["N", "dz"], fits=[])
    for n in a.sizes:
        if f"n{n}_e_max" not in ps:
            continue
        reps = min(a.max_reps, len(ps[f"n{n}_e_max"]))
        for r in range(reps):
            data = {h: ps[f"n{n}_{h}"][r].astype(float) for h in HEAD_SETS["+percrystal"]}
            for k, f in fitters.items():
                fr = f.fit(data)
                val, err = f.derived(fr)
                res["fits"].append(dict(n=n, rep=r, set=k, val=val.tolist(), err=err.tolist(),
                                        truth=truth.tolist() + [float(n), 0.0], ok=fr["ok"], at_bound=fr["at_bound"]))
        print(f"  n={n}: {reps} reps done ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(a.out + ".partial", "w"))
    os.rename(a.out + ".partial", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
