#!/usr/bin/env python3
"""Pack one gamma-library job's ROOT output into flat, memmap-able arrays.

  $NIDHOGG_PY pack_library.py <batch> <tag>[,<tag>...] [--force]

Reads  runs/<batch>/<tag>/out.root  + the job's cells file (from jobs.txt).
Writes packed/<batch>/<tag>/ with
  cells.npy        (n_cells, 3) float64   E0_MeV, z_cm, beta
  meta.json        events_per_cell, n_events, n_hits, source info
  hit_offset.npy   (n_events + 1,) uint32 event i's hits are [o[i], o[i+1])
  crystal.npy      (n_hits,) uint8        crystal ID 1-30
  edep.npy         (n_hits,) float32      MeV, summed per crystal, unsmeared
  cos_theta.npy    (n_events,) float16    emission direction in emitter frame
  phi.npy          (n_events,) float16
Event i belongs to cell i // events_per_cell (Ancalagon --gamma-library
convention). Includes events with no BGO hit: they matter for the
efficiency and for the recoil kick. Verifies counts against RunInfo and
writes DONE last, so a partial pack is never mistaken for a complete one.
"""
import argparse
import json
import os
import sys

import awkward as ak
import numpy as np
import uproot

DRAGON_NN = "/data/pagroup/pa02/padsley/dragon_nn"


def job_line(batch, tag):
    for line in open(os.path.join(DRAGON_NN, "runs", batch, "jobs.txt")):
        f = line.split()
        if f and f[0] == f"{batch}/{tag}":
            return f
    raise KeyError(tag)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("tags", help="comma-separated job tags")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    bad = 0
    for tag in a.tags.split(","):
        try:
            pack_one(a.batch, tag, a.force)
        except Exception as e:  # noqa: BLE001 -- report and continue with the rest
            print(f"FAILED {tag}: {e}", file=sys.stderr)
            bad += 1
    return 1 if bad else 0


def pack_one(batch, tag, force):
    out = os.path.join(DRAGON_NN, "packed", batch, tag)
    if os.path.exists(os.path.join(out, "DONE")) and not force:
        print("already packed", out)
        return
    f = job_line(batch, tag)
    mode, cells_path, per_cell = f[1], f[2], int(f[3])
    assert mode == "library", mode
    cells = np.atleast_2d(np.loadtxt(cells_path, comments="#"))
    src = os.path.join(DRAGON_NN, "runs", batch, tag, "out.root")

    with uproot.open(src) as r:
        ri = r["RunInfo"].arrays(library="np")
        n = int(ri["nEvents"][0])
        if n != len(cells) * per_cell or int(ri["eventsPerCell"][0]) != per_cell:
            raise ValueError(f"{src}: nEvents={n}, cells={len(cells)}, per_cell={per_cell}")
        ev = r["BgoEvent"].arrays(["eventID", "crystalID", "edepMeV"], library="ak")
        lt = r["LibTruth"].arrays(["cosThetaStar", "phiStar"], library="np")
    eid = ak.to_numpy(ev["eventID"])
    if len(eid) != n or not np.array_equal(eid, np.arange(n)):
        raise ValueError(f"{src}: BgoEvent eventIDs are not 0..n-1")
    if len(lt["cosThetaStar"]) != n:
        raise ValueError(f"{src}: LibTruth rows != nEvents")

    counts = ak.to_numpy(ak.num(ev["crystalID"])).astype(np.int64)
    hit_offset = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=hit_offset[1:])
    if hit_offset[-1] >= 2**32:
        raise ValueError("too many hits for uint32 offsets")
    crystal = ak.to_numpy(ak.flatten(ev["crystalID"])).astype(np.uint8)
    edep = ak.to_numpy(ak.flatten(ev["edepMeV"])).astype(np.float32)
    if crystal.size and (crystal.min() < 1 or crystal.max() > 30):
        raise ValueError("crystal ID outside 1-30")

    tmp = out + ".partial"
    os.makedirs(tmp, exist_ok=True)
    np.save(os.path.join(tmp, "cells.npy"), cells)
    np.save(os.path.join(tmp, "hit_offset.npy"), hit_offset.astype(np.uint32))
    np.save(os.path.join(tmp, "crystal.npy"), crystal)
    np.save(os.path.join(tmp, "edep.npy"), edep)
    np.save(os.path.join(tmp, "cos_theta.npy"), lt["cosThetaStar"].astype(np.float16))
    np.save(os.path.join(tmp, "phi.npy"), lt["phiStar"].astype(np.float16))
    meta = dict(batch=batch, tag=tag, events_per_cell=per_cell, n_events=n,
                n_hits=int(hit_offset[-1]), n_cells=len(cells), source=src,
                seed=int(ri["randomSeed"][0]), pressure_torr=float(ri["pressureTorr"][0]))
    with open(os.path.join(tmp, "meta.json"), "w") as fo:
        json.dump(meta, fo, indent=1)
    if os.path.exists(out):
        import shutil
        shutil.rmtree(out)
    os.rename(tmp, out)
    open(os.path.join(out, "DONE"), "w").close()
    print(f"packed {tag}: {n} events, {meta['n_hits']} hits, "
          f"{sum(os.path.getsize(os.path.join(out, x)) for x in os.listdir(out)) / 1e6:.0f} MB")


if __name__ == "__main__":
    sys.exit(main())
