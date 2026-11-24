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
