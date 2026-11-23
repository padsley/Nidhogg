#!/usr/bin/env python3
"""Write packed/<batch>/packlist.txt for pack.sub: the batch's library job
tags (from jobs.txt) in groups of --group, one comma-separated group per line.
Run on the cluster after the library jobs have finished and passed check_jobs.py."""
import argparse
import os

DRAGON_NN = "/data/pagroup/pa02/padsley/dragon_nn"
ap = argparse.ArgumentParser()
ap.add_argument("batch")
ap.add_argument("--group", type=int, default=17)
a = ap.parse_args()
tags = [l.split()[0].split("/", 1)[1] for l in open(os.path.join(DRAGON_NN, "runs", a.batch, "jobs.txt"))
        if l.strip() and l.split()[1] == "library"]
os.makedirs(os.path.join(DRAGON_NN, "packed", a.batch), exist_ok=True)
with open(os.path.join(DRAGON_NN, "packed", a.batch, "packlist.txt"), "w") as f:
    for i in range(0, len(tags), a.group):
        f.write(",".join(tags[i:i + a.group]) + "\n")
print(f"{len(tags)} tags in {(len(tags) + a.group - 1) // a.group} groups")
