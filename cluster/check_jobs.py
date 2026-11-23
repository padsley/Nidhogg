#!/usr/bin/env python3
"""Integrity check + cost summary for one batch, run ON THE CLUSTER with the
dragon_nn venv:

  $NIDHOGG_PY check_jobs.py <batch>   [--write-failed]

For every jobs.txt line:
  * status.txt exists with exit=0;
  * out.root opens and RunInfo.nEvents equals the requested events
    (reaction: NEVENTS; library: n_cells * events_per_cell);
  * BgoEvent rows == nEvents.
Summarises wall time and size per event by job kind and CPU model, peak
memory, and (reaction jobs) the DSSSD recoil fraction. With --write-failed
writes runs/<batch>/jobs_failed.txt (same format as jobs.txt) for
resubmission (done jobs are skipped by the wrapper anyway).
"""
import argparse
import collections
import os
import sys

import numpy as np
import uproot

DRAGON_NN = "/data/pagroup/pa02/padsley/dragon_nn"


def parse_status(path):
    d = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                d[k] = v
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--write-failed", action="store_true")
    a = ap.parse_args()
    bdir = os.path.join(DRAGON_NN, "runs", a.batch)
    lines = [l.split() for l in open(os.path.join(bdir, "jobs.txt")) if l.strip()]

    failed, rows = [], []
    for f in lines:
        name, mode, inp, nev = f[0], f[1], f[2], int(f[3])
        jdir = os.path.join(DRAGON_NN, "runs", name)
        st_path = os.path.join(jdir, "status.txt")
        problem = None
        st = parse_status(st_path) if os.path.exists(st_path) else {}
        if not st:
            problem = "no status (not run / still running)"
        elif st.get("exit") != "0":
            problem = f"exit={st.get('exit')}"
        else:
            try:
                want = nev * len(np.atleast_2d(np.loadtxt(inp, comments="#"))) if mode == "library" else nev
                with uproot.open(os.path.join(jdir, "out.root")) as r:
                    n = int(r["RunInfo"]["nEvents"].array(library="np")[0])
                    nrows = r["BgoEvent"].num_entries
                    rec = (float(r["BgoEvent"]["dsssdRecoil"].array(library="np").mean())
                           if mode == "reaction" else np.nan)
                if n != want or nrows != n:
                    problem = f"nEvents={n} rows={nrows} want={want}"
                else:
                    size = os.path.getsize(os.path.join(jdir, "out.root"))
                    kind = mode if mode == "library" else name.split("/")[1].rsplit("_P", 1)[0]
                    rows.append(dict(name=name, kind=kind, mode=mode, n=n, wall=float(st["wall_s"]),
                                     cpu=st.get("cpu", "?"), host=st.get("host", "?"),
                                     rss=int(st.get("maxrss_kb", 0)), size=size, recoil=rec))
            except Exception as e:  # noqa: BLE001
                problem = f"unreadable output: {e}"
        if problem:
            failed.append((" ".join(f), problem))

    print(f"batch {a.batch}: {len(lines)} jobs, {len(rows)} OK, {len(failed)} not OK")
    for line, why in failed:
        print(f"  NOT OK  {line.split()[0]}: {why}")
    if rows:
        print("\nper kind: jobs, events, wall-s/job (median), ms/event (median), bytes/event, max RSS MB, DSSSD recoil frac")
        by = collections.defaultdict(list)
        for r in rows:
            by[r["kind"]].append(r)
        for k, rs in sorted(by.items()):
            ms = [1e3 * r["wall"] / r["n"] for r in rs]
            print(f"  {k:24s} {len(rs):3d} {sum(r['n'] for r in rs):>11,d} "
                  f"{np.median([r['wall'] for r in rs]):8.0f} {np.median(ms):8.3f} "
                  f"{np.median([r['size'] / r['n'] for r in rs]):7.1f} "
                  f"{max(r['rss'] for r in rs) / 1024:7.0f} "
                  f"{np.nanmean([r['recoil'] for r in rs]):6.3f}")
        print("\nlibrary ms/event by CPU model:")
        bycpu = collections.defaultdict(list)
        for r in rows:
            if r["mode"] == "library":
                bycpu[r["cpu"]].append(1e3 * r["wall"] / r["n"])
        for c, v in sorted(bycpu.items(), key=lambda x: -len(x[1])):
            print(f"  {len(v):3d} jobs  median {np.median(v):.4f} ms/event  {c}")
        print(f"\ntotal output {sum(r['size'] for r in rows) / 1e9:.2f} GB, "
              f"{sum(r['wall'] for r in rows) / 3600:.2f} CPU-h")
    if a.write_failed:
        with open(os.path.join(bdir, "jobs_failed.txt"), "w") as fo:
            fo.write("".join(line + "\n" for line, _ in failed))
        print(f"wrote {len(failed)} lines to {bdir}/jobs_failed.txt")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
