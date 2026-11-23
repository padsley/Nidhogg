#!/usr/bin/env python3
"""Generate the production single-gamma deposit library batch ("lib_v1").

Grid (PLAN.md, measured costs from the pilot):
  E0    0.04 - 15.00 MeV in 20 keV steps   (749 energies)
  z     -8 ... +8 cm in 1 cm steps          (17; covers the cell and collars)
  beta  0.010 ... 0.050 in 0.005 steps      (9)
  N     40,000 events per cell
=> 114,597 cells, 4.58e9 events, ~285 CPU-h at 0.22 ms/event, ~185 GB ROOT.

One job per (z, beta, part): each (z, beta) slice's 749 energies are dealt
round-robin into PARTS jobs, so every job has the same energy mix (cost per
event rises with E0). Gas: H2 at 4 Torr. The few Torr of gas barely affect
the gamma response; the He cascades in the validation set test this.

Writes LOCAL_DIR/{jobs.txt, inputs/cells_*.txt, grid.npz}; rsync to
$DRAGON_NN/runs/lib_v1/ on the cluster.
"""
import argparse
import os
import shutil

import numpy as np

BATCH = "lib_v1"
CLUSTER_BATCH = f"/data/pagroup/pa02/padsley/dragon_nn/runs/{BATCH}"
ANC_REACTIONS = os.path.expanduser("~/codes/Ancalagon/reactions")

E_GRID = np.round(np.arange(0.04, 15.0 + 1e-9, 0.02), 5)
Z_GRID = np.arange(-8.0, 8.0 + 1e-9, 1.0)
BETA_GRID = np.round(np.arange(0.010, 0.050 + 1e-9, 0.005), 4)
EVENTS_PER_CELL = 40000
PARTS = 3
PRESSURE = 4.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{BATCH}_batch")
    a = ap.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    inp = os.path.join(a.out, "inputs")
    os.makedirs(inp)
    shutil.copy(os.path.join(ANC_REACTIONS, "k39pg_40ca.reaction"), os.path.join(inp, "gas_H2.reaction"))

    jobs = []
    seed = 2_000_000
    for iz, z in enumerate(Z_GRID):
        for ib, b in enumerate(BETA_GRID):
            for part in range(PARTS):
                es = E_GRID[part::PARTS]
                cells = np.column_stack([es, np.full_like(es, z), np.full_like(es, b)])
                tag = f"z{iz:02d}_b{ib}_p{part}"
                fn = f"cells_{tag}.txt"
                np.savetxt(os.path.join(inp, fn), cells, fmt="%.5f %.3f %.5f",
                           header=f"E0_MeV z_cm beta  (iz={iz} ib={ib} part={part})")
                seed += 1
                jobs.append(f"{BATCH}/{tag} library {CLUSTER_BATCH}/inputs/{fn} {EVENTS_PER_CELL} "
                            f"{seed} {PRESSURE} {CLUSTER_BATCH}/inputs/gas_H2.reaction -")
    with open(os.path.join(a.out, "jobs.txt"), "w") as f:
        f.write("\n".join(jobs) + "\n")
    np.savez(os.path.join(a.out, "grid.npz"), e=E_GRID, z=Z_GRID, beta=BETA_GRID,
             events_per_cell=EVENTS_PER_CELL, parts=PARTS)
    n_cells = len(E_GRID) * len(Z_GRID) * len(BETA_GRID)
    print(f"{len(jobs)} jobs, {n_cells} cells, {n_cells * EVENTS_PER_CELL:.3e} events "
          f"(~{n_cells * EVENTS_PER_CELL * 0.22e-3 / 3600:.0f} CPU-h, "
          f"~{n_cells * EVENTS_PER_CELL * 40 / 1e9:.0f} GB)")


if __name__ == "__main__":
    main()
