#!/usr/bin/env python3
"""Per-crystal effective Doppler factor table from the single-gamma library.

  python3 scripts/crystal_doppler_table.py PACKED_LIB OUT.npz [--e0 5.0 2.0 10.0] [--n 20000]

For every (z, beta) plane and each crystal c: over library events at E0 that
deposit > 50% of E0 in crystal c, the exact per-event Doppler factor
D = gamma*(1 + beta*cos(theta*)) (theta* = CM emission angle, stored in the
library) gives the mean <D>_c and spread sd(D)_c. Deposit-weighted mean over
the requested E0 values (interaction depth differs slightly with energy).
Saves z, beta, mean (nz, nb, 30), sd (nz, nb, 30), count.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nnemu.library import GammaLibrary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packed_lib")
    ap.add_argument("out")
    ap.add_argument("--e0", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    lib = GammaLibrary(a.packed_lib)
    zs, bs = lib.z_values, lib.beta_values
    s0 = np.zeros((len(zs), len(bs), 30))
    s1, s2 = np.zeros_like(s0), np.zeros_like(s0)
    rng = np.random.default_rng(0)
    for iz, z in enumerate(zs):
        for ib, b in enumerate(bs):
            es = lib.energies_at(z, b)
            g = 1 / np.sqrt(1 - b * b)
            for e0 in a.e0:
                e = es[np.abs(es - e0).argmin()]
                ev, cr, ed, cos, _ = lib.draw(e, z, b, a.n, rng)
                D = g * (1 + b * cos.astype(float))[ev]
                m = ed > 0.5 * e
                np.add.at(s0[iz, ib], cr[m] - 1, 1.0)
                np.add.at(s1[iz, ib], cr[m] - 1, D[m])
                np.add.at(s2[iz, ib], cr[m] - 1, D[m] ** 2)
        print(f"z {z:+.1f} done", flush=True)
    mean = s1 / np.maximum(s0, 1)
    sd = np.sqrt(np.maximum(s2 / np.maximum(s0, 1) - mean**2, 0))
    np.savez(a.out, z=zs, beta=bs, mean=mean, sd=sd, count=s0)
    print("min count per (z,b,c):", s0.min(), " D range", mean[s0 > 0].min(), mean.max())
    return 0


if __name__ == "__main__":
    sys.exit(main())
