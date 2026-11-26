"""Branching / yield / vertex fitter on top of the emulator.

Model for one run (one beam energy, pressure, gas):
    mu_head = N * sum_paths P(path | branchings) * sum_k w_k(dz) * T_head(path, z_k)
  * T_head(path, z_k): emulator probabilities per generated reaction at the
    vertex node z_k (ensemble mean; sd kept for an emulator-variance term),
    precomputed once per path on a fine z grid, so a fit is linear algebra.
  * w_k(dz): the run's vertex distribution p(z), shifted by a free dz.
  * P(path): products of per-level branching fractions, each level's vector a
    softmax of free logits (first fixed to 0).
  * N: total number of reactions (log-parametrised), i.e. yield / efficiency.
Per-crystal heads: the emulator gives RAW per-crystal deposit spectra; the
response's resolution is applied here as a matrix (exactly linear), so the
data side is per-crystal spectra of calibrated, smeared energies.
Likelihood: independent Poisson over the selected bins of the chosen heads.
(Heads share events; the closure tests measure what that does to pulls.)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import norm

from .model import encode_context
from .trainset import C_EDGES, E_EDGES, N_CRYSTALS

HEADS = ["e_max", "e_sum", "mult", "crystal_max", "per_crystal"]
EC = 0.5 * (E_EDGES[1:] + E_EDGES[:-1])
CC = 0.5 * (C_EDGES[1:] + C_EDGES[:-1])


# ------------------------------------------------------------- level scheme
class Scheme:
    """levels {id: Ex}, branches [(from, weight, to)], from = -1 is the
    entry (resonance) state at e_entry; level 0 is the ground state."""

    def __init__(self, levels: dict, branches: list, e_entry: float):
        self.ex = dict(levels)
        self.ex[-1] = e_entry
        self.ex.setdefault(0, 0.0)
        self.nodes = sorted({b[0] for b in branches}, key=lambda n: -self.ex[n])
        self.out = {n: [(to, w) for f, w, to in branches if f == n] for n in self.nodes}
        self.free_nodes = [n for n in self.nodes if len(self.out[n]) > 1]
        self.paths = []          # (gammas, [(node, branch index)])
        self._walk(-1, [], [])

    def _walk(self, node, gam, used):
        if node == 0:
            self.paths.append((sorted(gam, reverse=True), used))
            return
        for i, (to, _) in enumerate(self.out[node]):
            self._walk(to, gam + [self.ex[node] - self.ex[to]], used + [(node, i)])

    def true_fractions(self) -> dict:
        return {n: np.array([w for _, w in self.out[n]], float) / sum(w for _, w in self.out[n]) for n in self.nodes}

    def n_params(self) -> int:
        return sum(len(self.out[n]) - 1 for n in self.free_nodes)

    def fractions(self, theta: torch.Tensor) -> dict:
        fr, k = {}, 0
        for n in self.nodes:
            m = len(self.out[n])
            if m == 1:
                fr[n] = torch.ones(1, dtype=theta.dtype)
            else:
                fr[n] = torch.softmax(torch.cat([torch.zeros(1, dtype=theta.dtype), theta[k:k + m - 1]]), 0)
                k += m - 1
        return fr

    def logits_from_fractions(self, fr: dict) -> np.ndarray:
        return np.concatenate([np.log(fr[n][1:] / fr[n][0]) for n in self.free_nodes]) if self.free_nodes \
            else np.zeros(0)

    def path_probs(self, theta: torch.Tensor) -> torch.Tensor:
        fr = self.fractions(theta)
        return torch.stack([torch.prod(torch.stack([fr[n][i] for n, i in used])) for _, used in self.paths])


# ------------------------------------------------------------- templates
def smear_matrix(resp, crystal: int, nsub: int = 5) -> np.ndarray:
    """R[i, j]: P(observed bin i | raw deposit uniform in source bin j)."""
    w = C_EDGES[1] - C_EDGES[0]
    R = np.zeros((len(CC), len(CC)))
    eo = np.zeros((len(CC), N_CRYSTALS))
    eo[:, crystal] = CC
    thr = resp.threshold_eff(eo)[:, crystal]                     # at the observed energy
    for s in range(nsub):
        x = C_EDGES[:-1] + (s + 0.5) * w / nsub
        e = np.zeros((len(x), N_CRYSTALS))
        e[:, crystal] = x
        sig = np.maximum(resp.sigma(e)[:, crystal], 1e-4)
        cdf = norm.cdf((C_EDGES[:, None] - x[None, :]) / sig[None, :])
        R += np.diff(cdf, axis=0) * thr[:, None] / nsub
    return R


@torch.no_grad()
def emulate(models, gammas, z, beta, p0, acc, batch=32):
    """Ensemble mean and sd of head probabilities (without 'other') for
    rows of (gammas (n,6), z (n,)). Returns {head: (mean, sd)}."""
    n = len(z)
    ctx = encode_context(z, np.full(n, beta), np.full(n, p0), np.tile(acc, (n, 1)), np.ones(n), np.zeros(n))
    per = {h: [] for h in HEADS}
    for m in models:
        acc_h = {h: [] for h in HEADS}
        for i in range(0, n, batch):
            o = m(torch.from_numpy(gammas[i:i + batch]), torch.from_numpy(ctx[i:i + batch]))
            for h in HEADS:
                acc_h[h].append(torch.exp(o[h])[..., :-1].numpy())
        for h in HEADS:
            per[h].append(np.concatenate(acc_h[h]))
    return {h: (np.mean(v, 0), np.std(v, 0)) for h, v in per.items()}


class Templates:
    """T[head]: (n_paths, n_z, ...) emulator probabilities for a scheme at
    fixed beta, p0, acceptance, on z nodes; per_crystal is smeared to the
    observed-energy binning with the response."""

    def __init__(self, models, scheme: Scheme, z_nodes, beta, p0, acc, resp):
        self.z = np.asarray(z_nodes, float)
        nP, nZ = len(scheme.paths), len(self.z)
        g = np.zeros((nP * nZ, 6), np.float32)
        for p, (gam, _) in enumerate(scheme.paths):
            g[p * nZ:(p + 1) * nZ, :len(gam)] = gam
        out = emulate(models, g, np.tile(self.z, nP), beta, p0, acc)
        R = np.stack([smear_matrix(resp, c) for c in range(N_CRYSTALS)])        # (30, obs, raw)
        self.T, self.S = {}, {}
        for h, (mu, sd) in out.items():
            if h == "per_crystal":
                mu = np.einsum("cij,ncj->nci", R, mu)
                sd = np.sqrt(np.einsum("cij,ncj->nci", R**2, sd**2))
            self.T[h] = torch.tensor(mu.reshape(nP, nZ, *mu.shape[1:]), dtype=torch.float64)
            self.S[h] = torch.tensor(sd.reshape(nP, nZ, *sd.shape[1:]), dtype=torch.float64)


def default_bin_masks(e_min=0.5, pc_min=0.3) -> dict:
    return dict(e_max=torch.tensor(EC >= e_min), e_sum=torch.tensor(EC >= e_min),
                mult=None, crystal_max=None,
                per_crystal=torch.tensor(np.broadcast_to(CC >= pc_min, (N_CRYSTALS, len(CC))).copy()))


# ------------------------------------------------------------- fit
class Fitter:
    """Parameters: [branching logits..., log N, dz]."""

    def __init__(self, scheme: Scheme, templates: Templates, z_weights, heads, masks=None, fit_dz=True):
        self.sc, self.tp, self.heads = scheme, templates, list(heads)
        self.pz = torch.tensor(np.asarray(z_weights, float) / np.sum(z_weights), dtype=torch.float64)
        self.masks = default_bin_masks() if masks is None else masks
        self.fit_dz = fit_dz
        self.nb = scheme.n_params()

    def z_weights(self, dz):
        z = torch.tensor(self.tp.z, dtype=torch.float64)
        dzz = z[1] - z[0]
        f = ((z - dz - z[0]) / dzz).clamp(0, len(z) - 1 - 1e-9)
        i = f.floor().long()
        w = f - i
        p = self.pz[i] * (1 - w) + self.pz[(i + 1).clamp(max=len(z) - 1)] * w
        return p / p.sum()

    def mu(self, x):
        th, logN = x[:self.nb], x[self.nb]
        dz = x[self.nb + 1] if self.fit_dz else torch.zeros((), dtype=x.dtype)
        P = self.sc.path_probs(th)
        w = self.z_weights(dz)
        return {h: torch.exp(logN) * torch.einsum("p,z,pz...->...", P, w, self.tp.T[h]) for h in self.heads}

    def nll(self, x, data):
        mu, tot = self.mu(x), 0.0
        for h in self.heads:
            m, n = mu[h].clamp(min=1e-12), data[h]
            t = m - n * torch.log(m)
            msk = self.masks.get(h)
            tot = tot + (t[msk].sum() if msk is not None else t.sum())
        return tot

    def fit(self, data: dict, x0=None):
        data = {h: torch.as_tensor(np.asarray(data[h], float), dtype=torch.float64) for h in self.heads}
        npar = self.nb + 1 + int(self.fit_dz)
        if x0 is None:
            ref = data["mult"].sum() if "mult" in data else data[self.heads[0]].sum() / 0.5
            x0 = np.r_[np.zeros(self.nb), np.log(max(float(ref), 1.0)), np.zeros(int(self.fit_dz))]

        def f(xn):
            x = torch.tensor(xn, dtype=torch.float64, requires_grad=True)
            v = self.nll(x, data)
            v.backward()
            return float(v), x.grad.numpy().copy()

        bounds = [(-12, 12)] * self.nb + [(None, None)] + [(-4, 4)] * int(self.fit_dz)
        r = minimize(f, np.asarray(x0, float), jac=True, method="L-BFGS-B", bounds=bounds,
                     options=dict(maxiter=500))
        x = torch.tensor(r.x, dtype=torch.float64)
        H = torch.autograd.functional.hessian(lambda y: self.nll(y, data), x).numpy()
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = np.full((npar, npar), np.nan)
        return dict(x=r.x, cov=cov, nll=float(r.fun), ok=bool(r.success), at_bound=self._at_bound(r.x, bounds))

    @staticmethod
    def _at_bound(x, bounds):
        return any((lo is not None and xi <= lo + 1e-6) or (hi is not None and xi >= hi - 1e-6)
                   for xi, (lo, hi) in zip(x, bounds))

    def derived(self, res):
        """Branching fractions (flattened over free nodes), N, dz with
        delta-method errors."""
        x = torch.tensor(res["x"], dtype=torch.float64)

        def g(y):
            fr = self.sc.fractions(y[:self.nb])
            parts = [fr[n] for n in self.sc.free_nodes] + [torch.exp(y[self.nb:self.nb + 1])]
            if self.fit_dz:
                parts.append(y[self.nb + 1:])
            return torch.cat(parts)

        val = g(x).numpy()
        J = torch.autograd.functional.jacobian(g, x).numpy()
        err = np.sqrt(np.maximum(np.diag(J @ res["cov"] @ J.T), 0))
        return val, err
