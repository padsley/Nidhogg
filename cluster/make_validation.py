#!/usr/bin/env python3
"""Generate the full-cascade validation batch ("val_v1"): ground truth for
the factorization gate (library assembly vs full Ancalagon) and for the
end-to-end tests.

For each reaction in validation_reactions.REACTIONS:
  * pressures 1, 4, 10 Torr (H2 or He from the TARG card), T = 300 K;
  * two aimed mean vertex positions (z = -2.5, +2.5 cm), using BKIN from
    bkin_slopes.json (calibrate_bkin.py) and RWID = 0.1 keV;
  * the illustrative branching scheme at 1e6 events, plus 10 random linear
    chains (multiplicity 1-5) at 1e5 events each.
No RTUN: the separator is auto-retuned per run (fixed in Ancalagon 86da9f2
for recoils with A < 33).
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.expanduser("~/codes/Ancalagon/reactions"))
from reaction_writer import ReactionSpec, linear_chain  # noqa: E402
from validation_reactions import (N_RANDOM_CHAINS, PRESSURES, REACTIONS,  # noqa: E402
                                  VERTEX_TARGETS_CM, random_chains)

BATCH = "val_v1"
CLUSTER_BATCH = f"/data/pagroup/pa02/padsley/dragon_nn/runs/{BATCH}"
HALF = 5.315
N_SCHEME, N_CHAIN = 1_000_000, 100_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{BATCH}_batch")
    a = ap.parse_args()
    slopes = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bkin_slopes.json")))
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    inp = os.path.join(a.out, "inputs")
    os.makedirs(inp)
    jobs, manifest = [], []
    seed = 3_000_000
    for ir, (name, (b, t, r, q, er, levels, branches)) in enumerate(REACTIONS.items()):
        base = dict(beam=b, target=t, recoil=r, charge_state=q, eres_mev=er)
        scheme = ReactionSpec.from_dict({**base, "levels": {str(k): v for k, v in levels.items()},
                                         "branches": [list(x) for x in branches]})
        cascades = [("scheme", scheme, N_SCHEME)]
        for ic, g in enumerate(random_chains(scheme.e_budget_mev, N_RANDOM_CHAINS, seed=100 + ir)):
            cascades.append((f"chain{ic:02d}", linear_chain(base, g), N_CHAIN))
        e0 = slopes[name]["e_res_lab"]
        s4 = slopes[name]["slope_4torr"]
        for p in PRESSURES:
            for zt in VERTEX_TARGETS_CM:
                bkin = e0 + s4 * (p / 4.0) * (zt + HALF)
                for ctag, spec, n in cascades:
                    spec.bkin_mev, spec.rwid_mev, spec.rtun = round(bkin, 5), 1.0e-4, None
                    spec.comments = [f"NN validation val_v1: {name} {ctag}, P={p} Torr, aimed vertex z={zt} cm"]
                    tag = f"{name}_{ctag}_P{p:g}_z{zt:+.1f}"
                    fn = f"{tag}.reaction"
                    spec.write(os.path.join(inp, fn))
                    seed += 1
                    jobs.append(f"{BATCH}/{tag} reaction {CLUSTER_BATCH}/inputs/{fn} {n} {seed} {p} - -")
                    paths = [(pr, spec.gamma_energies(pa)) for pr, pa in spec.paths()]
                    manifest.append(dict(tag=tag, reaction=name, cascade=ctag, pressure=p, z_aim=zt,
                                         bkin=bkin, n_events=n, beta=spec.compound_beta(),
                                         paths=paths))
    with open(os.path.join(a.out, "jobs.txt"), "w") as f:
        f.write("\n".join(jobs) + "\n")
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=0)
    n_ev = sum(m["n_events"] for m in manifest)
    print(f"{len(jobs)} jobs, {n_ev:.3e} events (~{n_ev * 6.5e-3 / 3600:.0f} CPU-h at 6.5 ms/event)")


if __name__ == "__main__":
    main()
