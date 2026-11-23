#!/usr/bin/env python3
"""Phase-1 regression check: compact Ancalagon output vs the cached k39
spectra in ~/data/Ancalagon_k39_emulator/dataset.npz.

For each run directory named after a dataset stem (containing an
OUTPUT_MODE=compact dragon_hits.root), histogram singles and addback on the
dataset's own 20 keV edges and compare per-simulated-event spectra above
--emin. The comparison uses 100 keV rebinned bins with at least 5 counts.
Both are independent Poisson samples, so chi2/ndf should be about 1.

Below about 0.5 MeV the old extraction differs by construction: it counted
events whose only BGO steps had edep=0 as E=0 hits.

  python3 scripts/regression_k39.py RUN_DIR [RUN_DIR ...]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nnemu.ancalagon_io import read_compact  # noqa: E402

DATASET = os.path.expanduser("~/data/Ancalagon_k39_emulator/dataset.npz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--root-name", default="dragon_hits.root")
    ap.add_argument("--emin", type=float, default=0.5)
    ap.add_argument("--rebin", type=int, default=5)
    a = ap.parse_args()

    d = np.load(DATASET, allow_pickle=True)
    stems = list(d["file_stems"])
    edges = d["edges"]
    i0 = int(np.searchsorted(edges, a.emin))
    worst = 0.0
    for rd in a.run_dirs:
        stem = os.path.basename(os.path.normpath(rd))
        i = stems.index(stem)
        n_old = int(d["n_total"][i])
        run = read_compact(os.path.join(rd, a.root_name), truth=False)
        hit = run.hit_mask()
        for name, vals, key in [("singles", run.singles()[hit], "Y_singles"),
                                ("addback", run.addback()[hit], "Y_addback")]:
            new = np.histogram(vals, edges)[0].astype(float)
            old = np.rint(np.asarray(d[key][i], float) * n_old)
            m = (len(edges) - 1 - i0) // a.rebin * a.rebin
            x = new[i0:i0 + m].reshape(-1, a.rebin).sum(1)
            y = old[i0:i0 + m].reshape(-1, a.rebin).sum(1)
            ok = (x + y) >= 5
            A, B = x / run.n_events, y / n_old
            var = x / run.n_events ** 2 + y / n_old ** 2
            chi2 = float(((A - B)[ok] ** 2 / var[ok]).sum())
            ndf = int(ok.sum())
            worst = max(worst, chi2 / max(ndf, 1))
            print(f"{stem:45s} {name:7s} chi2/ndf = {chi2:6.1f}/{ndf:3d}  "
                  f"frac(E>{a.emin}) new={A.sum():.4f} old={B.sum():.4f}")
    print(f"worst chi2/ndf = {worst:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
