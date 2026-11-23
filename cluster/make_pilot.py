#!/usr/bin/env python3
"""Generate the Phase-2 pilot batch: inputs + jobs.txt.

Writes to LOCAL_DIR (default ./pilot_batch); rsync it to
$DRAGON_NN/runs/pilot/ on the cluster. Paths inside jobs.txt are cluster
paths.

Goals:
  * cost per event and bytes per event on the real pool (library and full
    cascades), memory use, node/CPU mix, failure modes;
  * a small but useful single-gamma library (H2: fine E grid at the k39
    vertex/beta, coarse grid at other z; He: coarse grid at other beta) for
    prototyping the assembler;
  * full-cascade runs with realistic BKIN vertex sampling at several
    pressures (k39 on H2, 22Ne(a,g) and 15O(a,g) on He) for first
    factorization checks;
  * two fixed-vertex k39 stems repeated from the local Phase-1 regression,
    to check the cluster build against the cached k39 dataset.
"""
import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/codes/Ancalagon/reactions"))
from reaction_writer import ReactionSpec, linear_chain  # noqa: E402

CLUSTER_BATCH = "/data/pagroup/pa02/padsley/dragon_nn/runs/pilot"
ANC_REACTIONS = os.path.expanduser("~/codes/Ancalagon/reactions")

K39 = dict(beam="39K", target="1H", recoil="40Ca", charge_state=8, eres_mev=0.606)
NE22 = dict(beam="22Ne", target="4He", recoil="26Mg", charge_state=6, eres_mev=0.83)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pilot_batch")
    a = ap.parse_args()
    out = a.out
    if os.path.exists(out):
        shutil.rmtree(out)
    inp = os.path.join(out, "inputs")
    os.makedirs(inp)
    cin = CLUSTER_BATCH + "/inputs"
    jobs = []
    seed = [1000]

    def job(name, mode, input_file, nevents, pressure, gasfile="-", extra="-"):
        seed[0] += 1
        jobs.append(f"pilot/{name} {mode} {cin}/{input_file} {nevents} {seed[0]} {pressure} "
                    f"{gasfile if gasfile == '-' else cin + '/' + gasfile} {extra}")

    # Gas-selector reaction files for library mode (only TARG matters).
    shutil.copy(os.path.join(ANC_REACTIONS, "k39pg_40ca.reaction"), os.path.join(inp, "gas_H2.reaction"))
    shutil.copy(os.path.join(ANC_REACTIONS, "o15ag_19ne.reaction"), os.path.join(inp, "gas_He.reaction"))

    k39_beta = ReactionSpec.from_dict({**K39, "branches": [[-1, 100, 0]]}).compound_beta()

    def cells_jobs(tag, cells, per_job, events_per_cell, gas):
        for j in range(0, len(cells), per_job):
            fn = f"cells_{tag}_{j // per_job:02d}.txt"
            np.savetxt(os.path.join(inp, fn), cells[j:j + per_job], fmt="%.5f %.3f %.6f",
                       header="E0_MeV z_cm beta")
            job(f"lib_{tag}_{j // per_job:02d}", "library", fn, events_per_cell, 4.0,
                gasfile=f"gas_{gas}.reaction")

    e_fine = np.round(np.arange(0.1, 15.0001, 0.1), 4)    # 150
    e_coarse = np.round(np.arange(0.5, 15.0001, 0.5), 4)  # 30
    L1 = np.array([(e, 0.0, k39_beta) for e in e_fine])
    L2 = np.array([(e, z, k39_beta) for z in (-8.0, -4.0, 4.0, 8.0) for e in e_coarse])
    L3 = np.array([(e, 0.0, b) for b in (0.01, 0.02, 0.05) for e in e_coarse])
    cells_jobs("H2_z0", L1, 15, 50000, "H2")     # 10 jobs x 750k events
    cells_jobs("H2_zscan", L2, 15, 50000, "H2")  # 8 jobs
    cells_jobs("He_bscan", L3, 15, 50000, "He")  # 6 jobs

    # Full cascades with BKIN vertex sampling, no RTUN (auto-retune per run).
    k39_bk = dict(**K39, bkin_mev=24.12, rwid_mev=1.0e-4)
    k39_cases = {
        "gs": linear_chain(k39_bk, [8.93377]),
        "2g": linear_chain(k39_bk, [8.93377 - 3.90438, 3.90438]),
        "3g": linear_chain(k39_bk, [8.93377 - 6.0, 6.0 - 2.2, 2.2]),
    }
    for tag, spec in k39_cases.items():
        spec.bkin_mev, spec.rwid_mev = 24.12, 1.0e-4
        fn = f"k39_{tag}_bkin.reaction"
        spec.write(os.path.join(inp, fn))
        for p in (1.0, 4.0, 10.0):
            job(f"rx_k39_{tag}_P{p:g}", "reaction", fn, 4000, p)
    branch = ReactionSpec.from_dict({**k39_bk, "levels": {"1": 3.90438, "2": 6.750},
                                     "branches": [[-1, 20, 0], [-1, 50, 1], [-1, 30, 2],
                                                  [2, 100, 1], [1, 100, 0]]})
    branch.write(os.path.join(inp, "k39_branch_bkin.reaction"))
    job("rx_k39_branch_P4", "reaction", "k39_branch_bkin.reaction", 10000, 4.0)

    ne22 = ReactionSpec.from_dict({**NE22, "levels": {"1": 1.80874},
                                   "branches": [[-1, 60, 0], [-1, 40, 1], [1, 100, 0]],
                                   "bkin_mev": 5.45, "rwid_mev": 1.0e-4})
    ne22.write(os.path.join(inp, "ne22ag_bkin.reaction"))
    for p in (2.0, 4.0, 8.0):
        job(f"rx_ne22ag_P{p:g}", "reaction", "ne22ag_bkin.reaction", 3000, p)
    shutil.copy(os.path.join(ANC_REACTIONS, "o15ag_19ne.reaction"), os.path.join(inp, "o15ag_19ne.reaction"))
    job("rx_o15ag_P4", "reaction", "o15ag_19ne.reaction", 3000, 4.0)

    # Cluster-build regression: same stems/event counts as the local check.
    # Legacy pressure handling (5 Torr, no temperature scaling) so they match
    # the cached dataset's conditions: GAS_TEMPERATURE_K is overridden empty.
    for stem, sub, n in [("k39pg_40ca_calib1g_d4450keV", "", 8000),
                         ("k39pg_40ca_cascade_45_4500keV", "", 50000)]:
        shutil.copy(os.path.join(ANC_REACTIONS, sub, stem + ".reaction"), os.path.join(inp, stem + ".reaction"))
        job(f"reg/{stem}", "reaction", stem + ".reaction", n, 5.0, extra="GAS_TEMPERATURE_K=")

    with open(os.path.join(out, "jobs.txt"), "w") as f:
        f.write("\n".join(jobs) + "\n")
    print(f"{len(jobs)} jobs written to {out}/jobs.txt; k39 compound beta = {k39_beta:.6f}")


if __name__ == "__main__":
    main()
