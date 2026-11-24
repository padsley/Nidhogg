#!/usr/bin/env python3
"""Recoil acceptance scans for the val_v1 conditions ("rscan_v1").

One condition per (reaction, pressure, aimed vertex z) of val_v1, reusing
that condition's scheme reaction file (only BEAM/TARG/RECL/ERES/BKIN/RWID
matter: RECOIL_SCAN drops the gammas). Each condition: JOBS_PER_COND jobs x
EVENTS_PER_JOB recoils, kicks uniform in |theta_x|,|theta_y| <= 30 mrad and
|delta| <= 3.5% (the separator accepts about +-20 mrad x +-2%; kicks outside
the box are treated as not transmitted).
"""
import argparse
import json
import os
import shutil

BATCH = "rscan_v1"
DRAGON_NN = "/data/pagroup/pa02/padsley/dragon_nn"
THETA_MAX, DELTA_MAX = 0.030, 0.035
JOBS_PER_COND, EVENTS_PER_JOB = 10, 100_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{BATCH}_batch")
    ap.add_argument("--val-manifest", default="val_v1_batch/manifest.json")
    a = ap.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    man = [m for m in json.load(open(a.val_manifest)) if m["cascade"] == "scheme"]
    jobs, conds = [], []
    seed = 4_000_000
    extra = f"RECOIL_SCAN=1,RECOIL_SCAN_THETA_MAX={THETA_MAX},RECOIL_SCAN_DELTA_MAX={DELTA_MAX}"
    for m in man:
        cond = m["tag"].replace("_scheme", "")
        conds.append(dict(cond=cond, reaction=m["reaction"], pressure=m["pressure"], z_aim=m["z_aim"],
                          val_tag=m["tag"], theta_max=THETA_MAX, delta_max=DELTA_MAX))
        for j in range(JOBS_PER_COND):
            seed += 1
            jobs.append(f"{BATCH}/{cond}/j{j:02d} reaction {DRAGON_NN}/runs/val_v1/inputs/{m['tag']}.reaction "
                        f"{EVENTS_PER_JOB} {seed} {m['pressure']} - {extra}")
    with open(os.path.join(a.out, "jobs.txt"), "w") as f:
        f.write("\n".join(jobs) + "\n")
    with open(os.path.join(a.out, "conditions.json"), "w") as f:
        json.dump(conds, f, indent=0)
    print(f"{len(conds)} conditions, {len(jobs)} jobs, {len(jobs) * EVENTS_PER_JOB:.2e} recoils")


if __name__ == "__main__":
    main()
