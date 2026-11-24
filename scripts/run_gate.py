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
from nnemu.observables import compare_counts, compare_weighted, histograms  # noqa: E402
from nnemu.recoil import (AcceptanceModel, assembled_recoil, condition_columns,  # noqa: E402
                          recoil_features)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cluster"))
from validation_reactions import REACTIONS  # noqa: E402

NUC = {"1H": (1, 1), "4He": (2, 4), "40Ca": (20, 40), "39Ca": (20, 39), "20Ne": (10, 20),
       "23Na": (11, 23), "26Mg": (12, 26), "19Ne": (10, 19), "39K": (19, 39), "38K": (19, 38),
       "19F": (9, 19), "22Ne": (10, 22), "15O": (8, 15)}
from nnemu.response import Response  # noqa: E402

E_EDGES = np.arange(0.0, 15.0001, 0.05)
C_EDGES = np.arange(0.0, 15.0001, 0.1)
HEADS = ["e_max", "e_sum", "mult", "crystal_max", "per_crystal"]


def gate_one(run_dir, lib, responses, max_draws, max_events, rng, acceptance=None, manifest=None):
    run = read_compact(os.path.join(run_dir, "out.root"))
    tr = run.truth
    n = min(run.n_events, max_events)
    k = max(1, max_draws // n)
    dep_full = run.dense_deposits()[:n]
    ge = [np.asarray(x, float) for x in ak.to_list(tr["gammaEStar"][:n])]
    z = ak.to_numpy(tr["vertexZ_cm"][:n]).astype(float)
    b = ak.to_numpy(tr["emitterBeta"][:n]).astype(float)
    reps = np.tile(np.arange(n), k)
    dep_asm, g_event, g_cos, g_phi, lib_f = assemble(lib, [ge[i] for i in reps], z[reps], b[reps], rng)
    gated = run.dsssd_recoil[:n].astype(bool)
    w_model = None
    if acceptance is not None:
        # p0 per real event from its truth (recoil + gammas), reused by its replicas
        el, cl, pl = tr["gammaELab"][:n], tr["gammaCosThetaLab"][:n], tr["gammaPhiLab"][:n]
        sl = np.sqrt(np.maximum(1 - cl**2, 0.0))
        p0 = np.sqrt((ak.to_numpy(tr["recoilPx"][:n]) + ak.to_numpy(ak.sum(el * sl * np.cos(pl), axis=1))) ** 2
                     + (ak.to_numpy(tr["recoilPy"][:n]) + ak.to_numpy(ak.sum(el * sl * np.sin(pl), axis=1))) ** 2
                     + (ak.to_numpy(tr["recoilPz"][:n]) + ak.to_numpy(ak.sum(el * cl, axis=1))) ** 2)
        g_e = np.concatenate([ge[i] for i in reps])
        rx, ry, rz = assembled_recoil(g_event, g_e, g_cos, g_phi, b[reps], p0[reps], len(reps))
        beam, targ, rec = REACTIONS[manifest["reaction"]][:3]
        Zr, Ar = NUC[rec]
        e_per_u = float(np.mean(ak.to_numpy(tr["beamKeMeV"][:n]))) / NUC[beam][1]
        X = np.hstack([recoil_features(rx, ry, rz, p0[reps], z[reps]),
                       condition_columns(len(reps), manifest["pressure"], Ar, Zr, e_per_u, targ == "4He")])
        w_model = acceptance.predict(X.astype(np.float32))
    del run, tr
    res = dict(run=os.path.basename(os.path.normpath(run_dir)), n=n, k=k, lib_var_factor=lib_f,
               z_mean=float(z.mean()), z_sd=float(z.std()), beta=float(b.mean()),
               recoil_frac=float(gated.mean()), heads={})
    if w_model is not None:
        res["recoil_frac_model"] = float(w_model.mean())
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
        if w_model is not None:
            # gated full events vs acceptance-weighted assembled events, both
            # normalised per GENERATED event (n and n*k)
            hf = histograms(df[gated], E_EDGES, C_EDGES)
            hw = histograms(da, E_EDGES, C_EDGES, weights=w_model)
            for head in HEADS:
                c0, ndf, _ = compare_weighted(hf[head], n, hw[head], hw[head + "_w2"], n * k)
                c1, _, _ = compare_weighted(hf[head], n, hw[head], hw[head + "_w2"], n * k,
                                            b_extra_var_factor=lib_f)
                res["heads"][f"{rname}/gated_model/{head}"] = dict(chi2_strict=c0, chi2_lib=c1, ndf=ndf)
            res["heads"][f"{rname}/gated_model/p_hit"] = dict(
                full=(hf["n_events"] - hf["n_nohit"]) / n, asm=(hw["n_events"] - hw["n_nohit"]) / (n * k))
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
    ap.add_argument("--acceptance", default=None, help="AcceptanceModel pickle: adds gated_model comparisons")
    ap.add_argument("--manifest", default=None, help="batch manifest.json (needed with --acceptance)")
    ap.add_argument("--tag", default="", help="suffix for output JSON names")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    lib = GammaLibrary(a.packed_lib)
    print(f"library: {len(lib.index)} cells, {len(lib.parts)} parts ({time.time() - t0:.0f} s)", flush=True)
    responses = [("raw", None), ("s1805", Response.load(a.response))]
    acc = AcceptanceModel.load(a.acceptance) if a.acceptance else None
    man = {m["tag"]: m for m in json.load(open(a.manifest))} if a.manifest else {}
    bad = 0
    for i, rd in enumerate(a.run_dirs):
        name = os.path.basename(os.path.normpath(rd))
        out = os.path.join(a.out_dir, name + a.tag + ".json")
        if os.path.exists(out):
            continue
        try:
            t1 = time.time()
            r = gate_one(rd, lib, responses, a.max_draws, a.max_events, np.random.default_rng(a.seed + i),
                         acceptance=acc, manifest=man.get(name))
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
