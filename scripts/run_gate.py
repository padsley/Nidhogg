#!/usr/bin/env python3
"""Factorization gate over many full-cascade runs (cluster job).

  python3 scripts/run_gate.py PACKED_LIB OUT_DIR RUN_DIR [RUN_DIR ...]
        [--response responses/s1805.yaml] [--max-draws 200000] [--max-events 200000]

For each RUN_DIR (containing out.root from an OUTPUT_MODE=compact reaction
run), assemble K = max(1, max_draws // n) events per real event from the
library, using that event's truth (gamma energies, vertex z, emitter beta).
Compare all heads, raw and after the response, both for all events and for
recoil-gated events. The gated assembled side is NOT yet gated: no recoil
acceptance model exists, so its chi2 is diagnostic only. Writes
OUT_DIR/<run name>.json with, per (response, head): chi2 and ndf, both
strict (no library term) and with the library-resampling term, plus P(hit)
and the recoil fraction.
"""
import argparse
import json
import os
import sys
import time

import awkward as ak
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nnemu.ancalagon_io import read_compact  # noqa: E402
from nnemu.assemble import assemble  # noqa: E402
from nnemu.library import GammaLibrary  # noqa: E402
from nnemu.observables import compare_counts, histograms  # noqa: E402
from nnemu.response import Response  # noqa: E402

E_EDGES = np.arange(0.0, 15.0001, 0.05)
C_EDGES = np.arange(0.0, 15.0001, 0.1)
HEADS = ["e_max", "e_sum", "mult", "crystal_max", "per_crystal"]


def gate_one(run_dir, lib, responses, max_draws, max_events, rng):
    run = read_compact(os.path.join(run_dir, "out.root"))
    tr = run.truth
    n = min(run.n_events, max_events)
    k = max(1, max_draws // n)
    dep_full = run.dense_deposits()[:n]
    ge = [np.asarray(x, float) for x in ak.to_list(tr["gammaEStar"][:n])]
    z = ak.to_numpy(tr["vertexZ_cm"][:n]).astype(float)
    b = ak.to_numpy(tr["emitterBeta"][:n]).astype(float)
    reps = np.tile(np.arange(n), k)
    dep_asm, _, _, _, lib_f = assemble(lib, [ge[i] for i in reps], z[reps], b[reps], rng)
    gated = run.dsssd_recoil[:n].astype(bool)
    del run, tr
    res = dict(run=os.path.basename(os.path.normpath(run_dir)), n=n, k=k, lib_var_factor=lib_f,
               z_mean=float(z.mean()), z_sd=float(z.std()), beta=float(b.mean()),
               recoil_frac=float(gated.mean()), heads={})
    for rname, resp in responses:
        df = dep_full if resp is None else resp.apply(dep_full, rng)
        da = dep_asm if resp is None else resp.apply(dep_asm, rng)
        for sel_name, sel_f, sel_a in [("all", slice(None), slice(None)),
                                       ("gated_full_only", gated, np.tile(gated, k))]:
            hf = histograms(df[sel_f], E_EDGES, C_EDGES)
            ha = histograms(da[sel_a], E_EDGES, C_EDGES)
            nf, na = hf["n_events"], ha["n_events"]
            for head in HEADS:
                c0, ndf, _ = compare_counts(hf[head], nf, ha[head], na)
                c1, _, _ = compare_counts(hf[head], nf, ha[head], na, b_extra_var_factor=lib_f)
                res["heads"][f"{rname}/{sel_name}/{head}"] = dict(chi2_strict=c0, chi2_lib=c1, ndf=ndf)
            res["heads"][f"{rname}/{sel_name}/p_hit"] = dict(
                full=1 - hf["n_nohit"] / nf, asm=1 - ha["n_nohit"] / na)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packed_lib")
    ap.add_argument("out_dir")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--response", default=os.path.join(os.path.dirname(__file__), "..", "responses", "s1805.yaml"))
    ap.add_argument("--max-draws", type=int, default=200_000)
    ap.add_argument("--max-events", type=int, default=200_000,
                    help="use at most this many events of each run (memory: slots are 1.4-1.8 GB)")
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    lib = GammaLibrary(a.packed_lib)
    print(f"library: {len(lib.index)} cells, {len(lib.parts)} parts ({time.time() - t0:.0f} s)", flush=True)
    responses = [("raw", None), ("s1805", Response.load(a.response))]
    bad = 0
    for i, rd in enumerate(a.run_dirs):
        name = os.path.basename(os.path.normpath(rd))
        out = os.path.join(a.out_dir, name + ".json")
        if os.path.exists(out):
            continue
        try:
            t1 = time.time()
            r = gate_one(rd, lib, responses, a.max_draws, a.max_events, np.random.default_rng(a.seed + i))
            r["seconds"] = time.time() - t1
            with open(out + ".partial", "w") as f:
                json.dump(r, f)
            os.rename(out + ".partial", out)
            h = r["heads"]
            print(f"{name}: n={r['n']} k={r['k']} libf={r['lib_var_factor']:.2f} "
                  + " ".join(f"{hd}={h['s1805/all/' + hd]['chi2_strict'] / max(h['s1805/all/' + hd]['ndf'], 1):.2f}"
                             for hd in HEADS) + f" ({r['seconds']:.0f} s)", flush=True)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAILED {name}: {e!r}", file=sys.stderr, flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
