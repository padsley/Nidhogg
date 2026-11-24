#!/usr/bin/env python3
"""Factorization gate: full Ancalagon cascades vs library assembly.

For a full-cascade compact output (with Truth), assemble K events per real
event from the single-gamma library, using that event's own gamma energies
(emitter frame), vertex z and emitter beta. Then compare every observable
head, raw and after a detector response:
  E_gamma1 (max crystal), sum, multiplicity, E_gamma1 crystal (hit pattern),
  and the 30 per-crystal spectra.
Prints chi2/ndf per head. Both sides are Poisson, and the assembled side is
scaled by 1/K. Library resampling noise is not included, so with K large
and a small library expect chi2/ndf slightly above 1 even if factorization
is exact.

  python3 scripts/factorization_check.py RUN.root PACKED_DIR [--k 4]
        [--response responses/s1805.yaml] [--z-grid 0] [--beta-grid 0.035458]
"""
import argparse
import os
import sys

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nnemu.ancalagon_io import read_compact  # noqa: E402
from nnemu.assemble import assemble  # noqa: E402
from nnemu.library import GammaLibrary  # noqa: E402
from nnemu.observables import compare_counts, histograms  # noqa: E402
from nnemu.response import Response  # noqa: E402

E_EDGES = np.arange(0.0, 15.0001, 0.05)
C_EDGES = np.arange(0.0, 15.0001, 0.1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("packed_dir")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--response", default=None, help="YAML; raw-only if omitted")
    ap.add_argument("--z-grid", type=float, nargs="*", default=None)
    ap.add_argument("--beta-grid", type=float, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--no-lib-noise", dest="lib_noise", action="store_false",
                    help="ignore library-resampling variance (default: include it, "
                         "computed from actual per-cell draw counts)")
    ap.add_argument("--lib-events", type=float, default=None,
                    help="override: assume a single cell of this size (factor 1 + n*K/lib_events)")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    run = read_compact(a.run_root)
    tr = run.truth
    n = run.n_events if a.max_events is None else min(run.n_events, a.max_events)
    dep_full = run.dense_deposits()[:n]
    ge = [np.asarray(x, float) for x in ak.to_list(tr["gammaEStar"][:n])]
    z = ak.to_numpy(tr["vertexZ_cm"][:n]).astype(float)
    b = ak.to_numpy(tr["emitterBeta"][:n]).astype(float)

    lib = GammaLibrary(a.packed_dir)
    reps = np.tile(np.arange(n), a.k)
    dep_asm, _, _, _, lib_f = assemble(lib, [ge[i] for i in reps], z[reps], b[reps], rng,
                           z_grid=a.z_grid, beta_grid=a.beta_grid)

    responses = [("raw", None)]
    if a.response:
        responses.append((os.path.basename(a.response), Response.load(a.response)))
    print(f"{os.path.basename(os.path.dirname(a.run_root)) or a.run_root}: {n} events, K={a.k}, "
          f"z {z.min():.2f}..{z.max():.2f}, beta {b.mean():.5f}, library variance factor {lib_f:.2f}")
    worst = 0.0
    for rname, resp in responses:
        if resp is None:
            hf, ha = histograms(dep_full, E_EDGES, C_EDGES), histograms(dep_asm, E_EDGES, C_EDGES)
        else:
            hf = histograms(resp.apply(dep_full, rng), E_EDGES, C_EDGES)
            ha = histograms(resp.apply(dep_asm, rng), E_EDGES, C_EDGES)
        line = []
        for head in ["e_max", "e_sum", "mult", "crystal_max", "per_crystal"]:
            f = (1.0 + n * a.k / a.lib_events) if a.lib_events else (lib_f if a.lib_noise else 1.0)
            chi2, ndf, pull = compare_counts(hf[head], n, ha[head], n * a.k, b_extra_var_factor=f)
            worst = max(worst, chi2 / max(ndf, 1))
            line.append(f"{head} {chi2:7.1f}/{ndf:<5d}")
        eff_f, eff_a = 1 - hf["n_nohit"] / n, 1 - ha["n_nohit"] / (n * a.k)
        print(f"  [{rname:10s}] " + "  ".join(line) + f"   P(hit) full {eff_f:.4f} asm {eff_a:.4f}")
    print(f"  worst chi2/ndf {worst:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
