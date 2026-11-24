#!/usr/bin/env python3
"""Summarise factorization-gate JSONs (scripts/run_gate.py).

  python3 scripts/summarize_gate.py GATE_DIR [--response s1805] [--sel all]

Prints, per head, the distribution of chi2/ndf over runs (strict, i.e.
without the library-resampling term, and with it). It then gives a per-reaction x
pressure table of strict chi2/ndf for the scalar heads and the per-crystal
spectra, the P(hit) agreement, and the runs failing the gate. A run fails
when, for some head, the p-value of the WITH-library chi2 is below 1e-3.
That variance term is conservative, so a failure there is a real
disagreement. The strict numbers bound the size of any problem from below.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
from scipy.stats import chi2 as chi2dist

HEADS = ["e_max", "e_sum", "mult", "crystal_max", "per_crystal"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate_dir")
    ap.add_argument("--response", default="s1805")
    ap.add_argument("--sel", default="all")
    ap.add_argument("--p-fail", type=float, default=1e-3)
    a = ap.parse_args()
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.gate_dir, "*.json")))]
    print(f"{len(rows)} runs, response={a.response}, selection={a.sel}\n")
    key = lambda h: f"{a.response}/{a.sel}/{h}"  # noqa: E731

    print("head          strict chi2/ndf: median  90%   max  | with-library: median  max")
    for h in HEADS:
        s = np.array([r["heads"][key(h)]["chi2_strict"] / max(r["heads"][key(h)]["ndf"], 1) for r in rows])
        l = np.array([r["heads"][key(h)]["chi2_lib"] / max(r["heads"][key(h)]["ndf"], 1) for r in rows])
        print(f"{h:13s} {np.median(s):26.2f} {np.percentile(s, 90):5.2f} {s.max():5.2f}  | {np.median(l):19.2f} {l.max():5.2f}")
    dp = np.array([r["heads"][key("p_hit")]["asm"] - r["heads"][key("p_hit")]["full"] for r in rows])
    print(f"P(hit) asm - full: mean {dp.mean():+.4f}, sd {dp.std():.4f}, max |.| {np.abs(dp).max():.4f}\n")

    def reac(r):
        return r["run"].split("_")[0]

    def pres(r):
        return [x for x in r["run"].split("_") if x.startswith("P")][0]

    tab = collections.defaultdict(list)
    for r in rows:
        tab[(reac(r), pres(r))].append(r)
    print("reaction P     n  " + "  ".join(f"{h:>11s}" for h in HEADS) + "   (strict chi2/ndf, pooled over runs)")
    for (rc, p), rs in sorted(tab.items()):
        cells = []
        for h in HEADS:
            c = sum(r["heads"][key(h)]["chi2_strict"] for r in rs)
            n = sum(r["heads"][key(h)]["ndf"] for r in rs)
            cells.append(f"{c / max(n, 1):11.2f}")
        print(f"{rc:8s} {p:4s} {len(rs):2d}  " + "  ".join(cells))

    print(f"\nruns failing (with-library p < {a.p_fail}):")
    nfail = 0
    for r in rows:
        bad = []
        for h in HEADS:
            d = r["heads"][key(h)]
            if d["ndf"] > 0 and chi2dist.sf(d["chi2_lib"], d["ndf"]) < a.p_fail:
                bad.append(f"{h} {d['chi2_lib']:.0f}/{d['ndf']} (strict {d['chi2_strict']:.0f})")
        if bad:
            nfail += 1
            print(f"  {r['run']}: z={r['z_mean']:.2f}+-{r['z_sd']:.2f} beta={r['beta']:.4f} libf={r['lib_var_factor']:.1f}: "
                  + "; ".join(bad))
    print(f"  {nfail} of {len(rows)} runs")


if __name__ == "__main__":
    main()
