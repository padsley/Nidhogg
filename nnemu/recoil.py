"""Recoil kinematics and separator acceptance for recoil-gated spectra.

The recoil's momentum at emission is the compound momentum p0 (along +z)
minus the sum of the gammas' lab momenta. Acceptance (reaching the DSSSD)
is modelled as a probability of event-level recoil features plus run
conditions, learned from full Ancalagon runs (their truth + dsssdRecoil):

  features:   theta_x = px/pz, theta_y = py/pz (rad), delta = |p|/p0 - 1,
              vertex z (cm)
  conditions: pressure (Torr), recoil A, recoil Z, beam energy per nucleon
              (MeV/u), gas (0 = H2, 1 = He)

The same feature function is applied to full-simulation truth and to
library-assembled events. That makes the gated factorization gate a direct
test of both the kick computation and the learned acceptance.
"""
from __future__ import annotations

import pickle

import numpy as np

AMU_MEV = 931.49432
FEATURES = ["theta_x", "theta_y", "delta", "z", "pressure", "A", "Z", "e_per_u", "gas_he"]


def gamma_lab_momenta(e_star, cos_star, phi_star, beta):
    """Lab momenta (MeV/c) of gammas emitted isotropically in a frame moving
    with speed beta along +z (all arrays broadcast)."""
    e_star, cos_star, phi_star, beta = map(np.asarray, (e_star, cos_star, phi_star, beta))
    g = 1.0 / np.sqrt(1.0 - beta**2)
    sin_star = np.sqrt(np.clip(1.0 - cos_star**2, 0.0, None))
    px = e_star * sin_star * np.cos(phi_star)
    py = e_star * sin_star * np.sin(phi_star)
    pz = g * e_star * (cos_star + beta)
    return px, py, pz


def compound_momentum(beta, mass_mev):
    beta = np.asarray(beta, float)
    return beta / np.sqrt(1.0 - beta**2) * mass_mev


def recoil_features(prx, pry, prz, p0, z):
    """Event-level features from the recoil momentum and compound momentum."""
    p = np.sqrt(prx**2 + pry**2 + prz**2)
    return np.column_stack([prx / prz, pry / prz, p / p0 - 1.0, z])


def assembled_recoil(gamma_event, gamma_e_star, gamma_cos, gamma_phi, beta_event, p0_event, n_events):
    """Recoil momentum per assembled event from its gammas (library
    emission angles). Gammas are boosted with the event's emitter beta.
    Returns (prx, pry, prz)."""
    b = beta_event[gamma_event]
    gx, gy, gz = gamma_lab_momenta(gamma_e_star, gamma_cos, gamma_phi, b)
    sx = np.bincount(gamma_event, gx, minlength=n_events)
    sy = np.bincount(gamma_event, gy, minlength=n_events)
    sz = np.bincount(gamma_event, gz, minlength=n_events)
    return -sx, -sy, p0_event - sz


def condition_columns(n, pressure, A, Z, e_per_u, gas_he):
    return np.column_stack([np.full(n, pressure), np.full(n, A), np.full(n, Z),
                            np.full(n, e_per_u), np.full(n, float(gas_he))])


class AcceptanceModel:
    """Gradient-boosted classifier P(recoil reaches DSSSD | features)."""

    def __init__(self, **kw):
        from sklearn.ensemble import HistGradientBoostingClassifier
        params = dict(max_iter=400, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
                      l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                      random_state=0)
        params.update(kw)
        self.clf = HistGradientBoostingClassifier(**params)

    def fit(self, X, y):
        self.clf.fit(X, y.astype(int))
        return self

    def predict(self, X) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "AcceptanceModel":
        with open(path, "rb") as f:
            return pickle.load(f)


class AcceptanceMap:
    """Acceptance for ONE experimental condition (reaction, pressure, beam
    energy, separator tune), learned from a recoil-only scan (Ancalagon
    RECOIL_SCAN): P(transmitted | theta_x, theta_y, delta, z). The scan
    samples kicks uniformly, so the sharp acceptance edges and corners are
    densely covered. Kicks outside the scanned box count as not transmitted.
    """

    def __init__(self, theta_max: float, delta_max: float, **kw):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.theta_max, self.delta_max = theta_max, delta_max
        params = dict(max_iter=600, learning_rate=0.15, max_leaf_nodes=127, min_samples_leaf=50,
                      max_bins=255, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                      random_state=0)
        params.update(kw)
        self.clf = HistGradientBoostingClassifier(**params)

    def fit(self, F, y):
        """F: (N,4) theta_x, theta_y, delta, z."""
        self.clf.fit(F, y.astype(int))
        return self

    def predict(self, F) -> np.ndarray:
        F = np.asarray(F, dtype=np.float32)
        inside = ((np.abs(F[:, 0]) <= self.theta_max) & (np.abs(F[:, 1]) <= self.theta_max)
                  & (np.abs(F[:, 2]) <= self.delta_max))
        p = np.zeros(len(F))
        if inside.any():
            p[inside] = self.clf.predict_proba(F[inside])[:, 1]
        return p


# --- Parametric acceptance (network input) -----------------------------
# Superellipse in (theta_x, theta_y), momentum edges quadratic in theta_x
# (the separator's chromatic coupling), logistic edges. 14 numbers, ordered
# as ACCEPTANCE_PARAMS. Angles in rad, delta as a fraction.
ACCEPTANCE_PARAMS = ["t0", "cx", "cy", "ax", "ay", "n", "wa",
                     "l0", "l1", "l2", "h0", "h1", "h2", "wd"]


def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def parametric_acceptance(params, theta_x, theta_y, delta):
    t0, cx, cy, ax, ay, n, wa, l0, l1, l2, h0, h1, h2, wd = params
    u = np.abs(theta_x - cx) / ax
    v = np.abs(theta_y - cy) / ay
    r = (u**n + v**n) ** (1.0 / n)
    tx = theta_x - cx
    dlo = l0 + l1 * tx + l2 * tx**2
    dhi = h0 + h1 * tx + h2 * tx**2
    return t0 * _sig((1.0 - r) / wa) * _sig((delta - dlo) / wd) * _sig((dhi - delta) / wd)


OPEN_ACCEPTANCE = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 1e-3,
                            -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1e-3])
_LO = np.array([0.5, -0.01, -0.01, 0.005, 0.005, 1.5, 0.005, -0.06, -2.0, -200.0, 0.0, -2.0, -200.0, 1e-5])
_HI = np.array([1.0, 0.01, 0.01, 0.05, 0.05, 12.0, 0.4, 0.0, 2.0, 200.0, 0.06, 2.0, 200.0, 0.01])
ACCEPTANCE_BOUNDS = (_LO, _HI)


def fit_parametric_acceptance(F, y, x0=None):
    """Maximum-likelihood fit to scan events (F columns theta_x, theta_y,
    delta[, z]; y transmitted). Returns (params, nll). Uses PyTorch
    (autograd + L-BFGS, bounds via a sigmoid reparametrisation) when
    available, otherwise scipy (slow: ~20 min for 2e5 events)."""
    try:
        import torch  # noqa: F401
        return _fit_torch(F, y, x0)
    except ImportError:
        return _fit_scipy(F, y, x0)


def _fit_torch(F, y, x0=None):
    import torch
    lo, hi = torch.tensor(_LO), torch.tensor(_HI)
    tx, ty, d = (torch.tensor(F[:, k], dtype=torch.float64) for k in range(3))
    yy = torch.tensor(y, dtype=torch.float64)
    if x0 is None:
        x0 = np.array([0.98, -0.0008, 0.0, 0.020, 0.020, 4.0, 0.05, -0.02, 0.0, 0.0, 0.021, 0.0, 0.0, 0.0015])

    def to_u(p):  # bounded -> unbounded
        f = (torch.tensor(p) - lo) / (hi - lo)
        f = f.clamp(1e-6, 1 - 1e-6)
        return torch.log(f / (1 - f))

    def to_p(u):
        return lo + (hi - lo) * torch.sigmoid(u)

    def nll_of(p):
        t0, cx, cy, ax, ay, n, wa, l0, l1, l2, h0, h1, h2, wd = p
        r = ((torch.abs(tx - cx) / ax) ** n + (torch.abs(ty - cy) / ay) ** n) ** (1.0 / n)
        txc = tx - cx
        dlo = l0 + l1 * txc + l2 * txc**2
        dhi = h0 + h1 * txc + h2 * txc**2
        t = t0 * torch.sigmoid((1 - r) / wa) * torch.sigmoid((d - dlo) / wd) * torch.sigmoid((dhi - d) / wd)
        t = t.clamp(1e-9, 1 - 1e-9)
        return -(yy * torch.log(t) + (1 - yy) * torch.log(1 - t)).mean()

    best_u, best = None, np.inf
    for n0 in (2.5, 4.0, 8.0):
        x = x0.copy(); x[5] = n0
        u = to_u(x).clone().requires_grad_(True)
        opt = torch.optim.LBFGS([u], lr=1.0, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12,
                                history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = nll_of(to_p(u))
            loss.backward()
            return loss
        for _ in range(4):
            opt.step(closure)
        with torch.no_grad():
            val = float(nll_of(to_p(u)))
        if val < best:
            best, best_u = val, u.detach().clone()
    return to_p(best_u).numpy(), best


def _fit_scipy(F, y, x0=None):
    from scipy.optimize import minimize
    tx, ty, d = (F[:, k].astype(float) for k in range(3))
    y = y.astype(float)
    if x0 is None:
        x0 = np.array([0.98, -0.0008, 0.0, 0.020, 0.020, 4.0, 0.05, -0.02, 0.0, 0.0, 0.021, 0.0, 0.0, 0.0015])

    def nll(p):
        t = np.clip(parametric_acceptance(p, tx, ty, d), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(t) + (1 - y) * np.log(1 - t))

    best = None
    for n0 in (2.5, 4.0, 8.0):
        x = x0.copy(); x[5] = n0
        r = minimize(nll, x, method="L-BFGS-B", bounds=list(zip(_LO, _HI)))
        if best is None or r.fun < best.fun:
            best = r
    r = minimize(nll, best.x, method="Powell", bounds=list(zip(_LO, _HI)),
                 options=dict(maxiter=20000, xtol=1e-6, ftol=1e-9))
    p = np.clip(r.x if r.fun < best.fun else best.x, _LO, _HI)
    return p, float(nll(p))


class ParametricAcceptanceMap:
    """Fitted parameters with the AcceptanceMap interface."""

    theta_max = None  # marks it as a per-condition map for run_gate

    def __init__(self, params):
        self.params = np.asarray(params, float)

    def predict(self, F):
        F = np.asarray(F)
        return parametric_acceptance(self.params, F[:, 0], F[:, 1], F[:, 2])
