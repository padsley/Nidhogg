"""Training-set generation for the NN emulator (Phase 5).

One sample = one (gamma path, vertex z, emitter beta, compound momentum p0,
acceptance window, response nuisances), with target histograms assembled
from the single-gamma library and normalised per GENERATED event (so
detection and gating efficiencies are part of the target):

  e_max       (N_E,)      E_gamma1 (highest crystal, after response)
  e_sum       (N_E,)      sum of crystals (after response)
  mult        (N_MULT,)   crystals above threshold, 0..N_MULT-1 (last = overflow)
  crystal_max (30,)       crystal of E_gamma1
  per_crystal (30, N_C)   RAW deposits per crystal (response applied
                          analytically at fit time, per crystal)
  p_pass                  sum of acceptance weights / N (gated fraction)
Every head is acceptance-weighted. An "open" window (OPEN_ACCEPTANCE) gives
ungated spectra.
"""
from __future__ import annotations

import numpy as np

from .assemble import assemble
from .observables import observables
from .recoil import (ACCEPTANCE_PARAMS, OPEN_ACCEPTANCE, AMU_MEV, assembled_recoil,
                     parametric_acceptance, recoil_features)

E_EDGES = np.round(np.arange(0.0, 15.0 + 1e-9, 0.05), 4)        # 300 bins
C_EDGES = np.round(np.arange(0.0, 15.0 + 1e-9, 0.10), 4)        # 150 bins
N_MULT = 9
MAX_GAMMAS = 6
N_CRYSTALS = 30

# Sampling ranges. Acceptance parameters (nnemu.recoil.ACCEPTANCE_PARAMS,
# superellipse + quadratic momentum edges) bracket the fits to the 36 val_v2
# recoil scans (recoil/pmaps_v2), broadened so other tunes fall inside.
_ACC_LO = np.array([0.90, -0.002, -0.002, 0.014, 0.014, 1.8, 0.005, -0.030, -0.10, -5.0, 0.010, -0.30, -5.0, 2e-4])
_ACC_HI = np.array([1.00, 0.002, 0.002, 0.028, 0.028, 4.0, 0.200, -0.010, 0.40, 35.0, 0.030, 0.10, 30.0, 3e-3])
_ACC_LOG = np.array([False] * 6 + [True] + [False] * 6 + [True])   # log-uniform: wa, wd
RANGES = dict(
    z=(-8.0, 8.0), beta=(0.010, 0.050), recoil_A=(12.0, 45.0),
    e_total=(0.1, 15.0), e_min_gamma=0.05,
    res_scale=(0.7, 1.3), thr_shift=(-0.05, 0.10),
    acc_lo=_ACC_LO, acc_hi=_ACC_HI, acc_log=_ACC_LOG,
)
MULT_WEIGHTS = np.array([0.25, 0.22, 0.20, 0.15, 0.11, 0.07])
P_OPEN = 0.3


def sample_acceptance(rng, n):
    lo, hi, lg = RANGES["acc_lo"], RANGES["acc_hi"], RANGES["acc_log"]
    u = rng.random((n, len(lo)))
    lin = lo + (hi - lo) * u
    log = np.exp(np.log(lo[lg]) + (np.log(hi[lg]) - np.log(lo[lg])) * u[:, lg])
    acc = lin.copy()
    acc[:, lg] = log
    acc[rng.random(n) < P_OPEN] = OPEN_ACCEPTANCE
    return acc


def sample_conditions(rng: np.random.Generator, n: int) -> dict:
    R = RANGES
    m = rng.choice(np.arange(1, MAX_GAMMAS + 1), size=n, p=MULT_WEIGHTS / MULT_WEIGHTS.sum())
    gammas = np.zeros((n, MAX_GAMMAS))
    for i in range(n):
        lo = max(R["e_total"][0], R["e_min_gamma"] * m[i] * 1.5)
        # 70% uniform in [1, 15] MeV (typical DRAGON cascade budgets), 30%
        # log-uniform from lo (dense coverage of low-energy paths)
        if rng.random() < 0.7:
            etot = rng.uniform(max(1.0, lo), R["e_total"][1])
        else:
            etot = np.exp(rng.uniform(np.log(lo), np.log(R["e_total"][1])))
        while True:
            g = np.diff(np.r_[0.0, np.sort(rng.uniform(0, etot, m[i] - 1)), etot])
            if (g >= R["e_min_gamma"]).all():
                break
        gammas[i, :m[i]] = np.sort(g)[::-1]
    beta = rng.uniform(*R["beta"], n)
    A = rng.uniform(*R["recoil_A"], n)
    p0 = beta / np.sqrt(1 - beta**2) * A * AMU_MEV
    acc = sample_acceptance(rng, n)
    return dict(mult=m, gammas=gammas, z=rng.uniform(*R["z"], n), beta=beta, p0=p0, recoil_A=A,
                acceptance=acc, res_scale=rng.uniform(*R["res_scale"], n),
                thr_shift=rng.uniform(*R["thr_shift"], n))


def make_targets(lib, cond: dict, i: int, n_events: int, response, rng) -> dict:
    """Assemble n_events for sample i of `cond` and return its target heads."""
    m = int(cond["mult"][i])
    g = cond["gammas"][i, :m]
    z = np.full(n_events, cond["z"][i])
    b = np.full(n_events, cond["beta"][i])
    dep, g_event, g_cos, g_phi, lib_f = assemble(lib, [g] * n_events, z, b, rng)
    p0 = np.full(n_events, cond["p0"][i])
    rx, ry, rz = assembled_recoil(g_event, np.tile(g, n_events), g_cos, g_phi, b, p0, n_events)
    F = recoil_features(rx, ry, rz, p0, z)
    w = parametric_acceptance(cond["acceptance"][i], F[:, 0], F[:, 1], F[:, 2])
    dr = response.apply(dep, rng, resolution_scale=cond["res_scale"][i], threshold_shift=cond["thr_shift"][i])
    o = observables(dr)
    h = o["hit"]
    out = dict(
        e_max=np.histogram(o["e_max"][h], E_EDGES, weights=w[h])[0],
        e_sum=np.histogram(o["e_sum"][h], E_EDGES, weights=w[h])[0],
        mult=np.bincount(np.minimum(o["mult"], N_MULT - 1), w, minlength=N_MULT),
        crystal_max=np.bincount(o["crystal_max"], w, minlength=N_CRYSTALS + 1)[1:],
        p_pass=np.array([w.sum()]),
    )
    pc = np.zeros((N_CRYSTALS, len(C_EDGES) - 1))
    for c in range(N_CRYSTALS):
        v = dep[:, c]
        s = v > 0
        pc[c] = np.histogram(v[s], C_EDGES, weights=w[s])[0]
    out["per_crystal"] = pc
    for k in out:
        out[k] = (out[k] / n_events).astype(np.float32)
    out["lib_var_factor"] = np.float32(lib_f)
    return out
