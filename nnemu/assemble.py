"""Event-level assembly of cascade events from the single-gamma library.

For each assembled event (one cascade path at vertex z, emitter speed beta),
one library event is drawn per gamma and the 30-crystal deposit vectors are
added crystal by crystal. Detector response, thresholds and observables are
applied afterwards, to the summed vector, so summing in a crystal and
max-crystal selection are treated exactly. This assumes the showers of
different gammas in one event are independent.

Off-grid values:
  * z, beta: stochastic linear interpolation between the two neighbouring
    grid planes, chosen per event (all gammas of an event share the plane).
  * E: nearest grid energy at that plane, with every deposit of that gamma
    scaled by E / E_grid (|E - E_grid| <= half a grid step, i.e. <= 10 keV
    for lib_v1).
"""
from __future__ import annotations

import numpy as np

from .library import GammaLibrary

N_CRYSTALS = 30


def _stochastic_plane(x: np.ndarray, grid: np.ndarray, rng) -> np.ndarray:
    """Pick grid[k] or grid[k+1] with linear weights; values outside the
    grid are clipped to its ends."""
    x = np.clip(x, grid[0], grid[-1])
    k = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, len(grid) - 2) if len(grid) > 1 else np.zeros(len(x), int)
    if len(grid) == 1:
        return np.full(len(x), grid[0])
    lo, hi = grid[k], grid[k + 1]
    w = (x - lo) / (hi - lo)
    return np.where(rng.random(len(x)) < w, hi, lo)


def assemble(lib: GammaLibrary, gamma_e: list, z: np.ndarray, beta: np.ndarray,
             rng: np.random.Generator, z_grid=None, beta_grid=None):
    """Assemble len(gamma_e) events.

    gamma_e: sequence (length N) of 1-D arrays of gamma energies in the
             emitter frame (MeV) for each event.
    z, beta: (N,) vertex z (cm) and emitter speed per event.
    z_grid/beta_grid: planes to interpolate between (default: all library
             planes). Restrict when the library is not a full product grid.
    Returns (deposits (N,30) float32 summed per crystal, gamma_event (M,),
    gamma_cos_theta (M,), gamma_phi (M,)): the last three give the emission
    direction of each assembled gamma (for the recoil kick).
    """
    n = len(gamma_e)
    z_grid = lib.z_values if z_grid is None else np.asarray(z_grid, float)
    beta_grid = lib.beta_values if beta_grid is None else np.asarray(beta_grid, float)
    zp = _stochastic_plane(np.asarray(z, float), z_grid, rng)
    bp = _stochastic_plane(np.asarray(beta, float), beta_grid, rng)

    mult = np.array([len(g) for g in gamma_e])
    g_event = np.repeat(np.arange(n), mult)
    g_e = np.concatenate([np.asarray(g, float) for g in gamma_e]) if n else np.zeros(0)
    g_z, g_b = zp[g_event], bp[g_event]

    # nearest grid energy per (plane) -- grids may differ between planes
    g_ecell = np.empty_like(g_e)
    planes = {}
    for key in set(zip(g_z.tolist(), g_b.tolist())):
        planes[key] = lib.energies_at(*key)
        if len(planes[key]) == 0:
            raise KeyError(f"library has no cells at z={key[0]}, beta={key[1]}")
    for (zz, bb), es in planes.items():
        m = (g_z == zz) & (g_b == bb)
        k = np.clip(np.searchsorted(es, g_e[m]), 1, len(es) - 1)
        nearer_lo = (g_e[m] - es[k - 1]) < (es[k] - g_e[m])
        g_ecell[m] = np.where(nearer_lo, es[k - 1], es[k]) if len(es) > 1 else es[0]
    scale = g_e / g_ecell

    dep = np.zeros((n, N_CRYSTALS), dtype=np.float32)
    g_cos = np.empty(len(g_e), np.float32)
    g_phi = np.empty(len(g_e), np.float32)
    keys = np.stack([g_ecell, g_z, g_b], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(uniq)))]
    for u, (e, zz, bb) in enumerate(uniq):
        gi = order[bounds[u]:bounds[u + 1]]
        draw, cry, ed, cs, ph = lib.draw(e, zz, bb, len(gi), rng)
        g_cos[gi], g_phi[gi] = cs, ph
        np.add.at(dep, (g_event[gi][draw], cry.astype(np.int64) - 1), ed * scale[gi][draw])
    return dep, g_event, g_cos, g_phi
