#!/usr/bin/env python3
"""Summarise closure fits (scripts/closure.py outputs).

  python3 scripts/summarize_closure.py CLOSURE_DIR [--by-reaction]

Pull = (fit - truth) / fit error. For each head set and sample size:
pull mean (bias in units of sigma) and sd (should be ~1) for branchings,
N and dz; median absolute error (sigma) of each; fraction of failed or
at-bound fits. Branching fractions of a node are not independent (they sum
to 1); all are pooled, which is fine for mean/sd of pulls.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

SETS = ["E1", "E1+sum", "+mult", "+hitpat", "+percrystal"]


def rob(p):
    """median pull / robust width (1.4826 MAD) / 68% coverage."""
    med = np.median(p)
    return f"{med:+5.2f}/{1.4826 * np.median(np.abs(p - med)):4.2f}/{np.mean(np.abs(p) < 1):3.0%}"


def summarize(fits, labels_by_run, title):
    print(f"\n### {title}")
    print(f"{'set':12s} {'n':>6s} | {'branch med/w/cov':>17s} {'med sig':>7s} | {'N med/w/cov':>15s} "
          f"{'sigN/N':>6s} | {'dz med/w/cov':>15s} {'sig dz':>6s} | bad")
    for s in SETS:
        for n in sorted({f["n"] for f in fits}):
            sel = [f for f in fits if f["set"] == s and f["n"] == n]
            if not sel:
                continue
            bp, bs, Np, Ns, zp, zs, bad = [], [], [], [], [], [], 0
            for f in sel:
                v, e, t = map(np.asarray, (f["val"], f["err"], f["truth"]))
                if not f["ok"] or f["at_bound"] or not np.all(np.isfinite(e)) or np.any(e[:-2] <= 0):
                    bad += 1
                    continue
                nb = len(v) - 2
                bp += list((v[:nb] - t[:nb]) / e[:nb])
                bs += list(e[:nb])
                Np.append((v[nb] - t[nb]) / e[nb])
                Ns.append(e[nb] / t[nb])
                zp.append(v[nb + 1] / e[nb + 1])
                zs.append(e[nb + 1])
            if not Np:
                continue
            bp, Np, zp = map(np.asarray, (bp, Np, zp))
            print(f"{s:12s} {n:6d} | {rob(bp)}  {np.median(bs):7.4f} | {rob(Np)} {np.median(Ns):6.3f} | "
                  f"{rob(zp)} {np.median(zs):6.2f} | {bad / len(sel):.0%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("closure_dir")
    ap.add_argument("--by-reaction", action="store_true")
    a = ap.parse_args()
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.closure_dir, "*.json")))]
    allf, byrx = [], defaultdict(list)
    for r in runs:
        for f in r["fits"]:
            f["run"] = r["run"]
        allf += r["fits"]
        byrx[r["run"].split("_")[0]] += r["fits"]
    summarize(allf, None, f"all {len(runs)} runs")
    if a.by_reaction:
        for rx, f in sorted(byrx.items()):
            summarize(f, None, rx)
    return 0


if __name__ == "__main__":
    os.sys.exit(main())
