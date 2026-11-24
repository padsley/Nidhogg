"""Observables from per-event 30-crystal deposit matrices (raw or after
detector response). Rows with no deposit above zero count as 'no hit'."""
from __future__ import annotations

import numpy as np

N_CRYSTALS = 30


def dense_from_compact(run) -> np.ndarray:
    """(N,30) float32 deposits from an ancalagon_io.CompactRun."""
    return run.dense_deposits()


def observables(dep: np.ndarray) -> dict:
    hit = dep > 0
    nhit = hit.sum(1)
    emax = dep.max(1)
    cmax = np.where(nhit > 0, dep.argmax(1) + 1, 0)
    return dict(
        e_max=emax,                    # E_gamma1: highest single-crystal energy
        e_sum=dep.sum(1),              # addback / sum over crystals
        mult=nhit,                     # crystals with energy
        crystal_max=cmax,              # hit pattern of the E_gamma1 crystal (0 = none)
        hit=nhit > 0,
    )


def histograms(dep: np.ndarray, e_edges: np.ndarray, crystal_e_edges: np.ndarray | None = None,
               max_mult: int = 10, weights: np.ndarray | None = None) -> dict:
    """Counts (or weighted sums, with per-event `weights`) for every head.
    Energy heads exclude no-hit events; their count is in 'n_nohit'. With
    weights, each head also gets '<head>_w2' (sum of squared weights, for
    the variance), and 'n_events' is the sum of weights."""
    o = observables(dep)
    h = o["hit"]
    w = np.ones(len(dep)) if weights is None else np.asarray(weights, float)

    def hist(x, edges, ww):
        return np.histogram(x, edges, weights=ww)[0]

    out = {}
    for ww, suffix in ([(w, "")] if weights is None else [(w, ""), (w**2, "_w2")]):
        out["e_max" + suffix] = hist(o["e_max"][h], e_edges, ww[h])
        out["e_sum" + suffix] = hist(o["e_sum"][h], e_edges, ww[h])
        out["mult" + suffix] = np.bincount(np.minimum(o["mult"], max_mult), ww, minlength=max_mult + 1)
        out["crystal_max" + suffix] = np.bincount(o["crystal_max"], ww, minlength=N_CRYSTALS + 1)[1:]
        ce = e_edges if crystal_e_edges is None else crystal_e_edges
        pc = np.zeros((N_CRYSTALS, len(ce) - 1))
        for c in range(N_CRYSTALS):
            v = dep[:, c]
            pc[c] = hist(v[v > 0], ce, ww[v > 0])
        out["per_crystal" + suffix] = pc
    if weights is None:
        for k in ("e_max", "e_sum", "mult", "crystal_max", "per_crystal"):
            out[k] = out[k].astype(np.int64)
    out["n_nohit"] = float(w[~h].sum())
    out["n_events"] = float(w.sum()) if weights is not None else len(dep)
    return out


def compare_weighted(a: np.ndarray, na: float, bw: np.ndarray, bw2: np.ndarray, nb: float,
                     min_counts: float = 5, b_extra_var_factor: float = 1.0):
    """chi2 between an unweighted Poisson histogram a (na events) and a
    weighted histogram (sums bw, squared sums bw2, nb generated events). It
    is the pooled analogue of compare_counts: per-event rate mu = mean of the
    two estimates, var(a/na) = mu/na, var(bw/nb) = mu * (bw2/bw) / nb."""
    a = np.asarray(a, float).ravel()
    bw = np.asarray(bw, float).ravel()
    bw2 = np.asarray(bw2, float).ravel()
    ok = (a + bw) >= min_counts
    mu = 0.5 * (a / na + bw / nb)
    weff = np.where(bw > 0, bw2 / np.where(bw > 0, bw, 1), 1.0)
    var = mu / na + b_extra_var_factor * mu * weff / nb
    pull = np.full(a.shape, np.nan)
    pull[ok] = (a[ok] / na - bw[ok] / nb) / np.sqrt(var[ok])
    return float(np.nansum(pull**2)), int(ok.sum()), pull


def compare_counts(a: np.ndarray, na: float, b: np.ndarray, nb: float, min_counts: int = 5,
                   b_extra_var_factor: float = 1.0):
    """Two-sample chi2 between Poisson histograms a (from na events) and b
    (from nb events), using the pooled estimate p = (a+b)/(na+nb) for the
    variance. Using each bin's own counts would bias the pulls negative in
    low-count bins. b_extra_var_factor > 1 inflates b's variance, e.g.
    1 + nb/n_library when b was resampled from a finite library.
    Bins with a+b < min_counts are skipped.
    Returns (chi2, ndf, pulls array (NaN where skipped))."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    ok = (a + b) >= min_counts
    p = (a + b) / (na + nb)
    var = p * (1.0 / na + b_extra_var_factor / nb)
    pull = np.full(a.shape, np.nan)
    pull[ok] = (a[ok] / na - b[ok] / nb) / np.sqrt(var[ok])
    return float(np.nansum(pull**2)), int(ok.sum()), pull
