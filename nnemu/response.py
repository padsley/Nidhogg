"""Selectable detector response (resolution, per-crystal threshold, gain),
applied to raw per-crystal deposits from Ancalagon or from assembly.

Response files (YAML, responses/*.yaml) hold per-crystal arrays; a scalar
is broadcast to all 30 crystals:

    name: s1805
    resolution:            # sigma(E)^2 = a^2 * E + b^2 * E^2 + c^2   (MeV)
      a: 0.0737            # stochastic term
      b: 0.0
      c: 0.0
    threshold:             # per-crystal acceptance = 1/(1+exp(-(E-e0)/width))
      e0: 0.1
      width: 0.01
    gain: 1.0              # E_measured = gain * E_deposited, before smearing

Doppler broadening is NOT part of the response: Ancalagon simulates it.
Analysis cuts on derived quantities (e.g. E_gamma1 > 1.4 MeV) belong to the
fit/observable definition, not here.
"""
from __future__ import annotations

import hashlib

import numpy as np
import yaml

N_CRYSTALS = 30


def _per_crystal(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    if a.ndim == 0:
        return np.full(N_CRYSTALS, float(a))
    if a.shape != (N_CRYSTALS,):
        raise ValueError(f"per-crystal value must be scalar or length {N_CRYSTALS}")
    return a


class Response:
    def __init__(self, spec: dict, source: str = "<dict>"):
        self.spec = spec
        self.name = spec.get("name", source)
        r, t = spec.get("resolution", {}), spec.get("threshold", {})
        self.a, self.b, self.c = (_per_crystal(r.get(k, 0.0)) for k in "abc")
        self.thr_e0 = _per_crystal(t.get("e0", 0.0))
        self.thr_w = _per_crystal(t.get("width", 0.0))
        self.gain = _per_crystal(spec.get("gain", 1.0))
        self.hash = hashlib.sha256(yaml.safe_dump(spec, sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def load(cls, path: str) -> "Response":
        with open(path) as f:
            return cls(yaml.safe_load(f), source=path)

    @classmethod
    def none(cls) -> "Response":
        return cls({"name": "none"})

    def sigma(self, e: np.ndarray) -> np.ndarray:
        """sigma (MeV) for an (N,30) energy matrix."""
        e = np.clip(e, 0, None)
        return np.sqrt(self.a**2 * e + self.b**2 * e**2 + self.c**2)

    def threshold_eff(self, e: np.ndarray) -> np.ndarray:
        w = np.where(self.thr_w > 0, self.thr_w, 1e-9)
        return 1.0 / (1.0 + np.exp(-np.clip((e - self.thr_e0) / w, -60, 60)))

    def apply(self, dep: np.ndarray, rng: np.random.Generator,
              resolution_scale: float = 1.0, threshold_shift: float = 0.0) -> np.ndarray:
        """Smear and threshold an (N,30) raw-deposit matrix. Crystals with no
        deposit stay 0; crystals failing the threshold are set to 0. The two
        keyword nuisances (global resolution scale, threshold shift in MeV)
        are what the emulator takes as inputs."""
        hit = dep > 0
        e = dep * self.gain
        s = self.sigma(e) * resolution_scale
        e = np.where(hit, e + s * rng.standard_normal(e.shape), 0.0)
        w = np.where(self.thr_w > 0, self.thr_w, 1e-9)
        p = 1.0 / (1.0 + np.exp(-np.clip((e - self.thr_e0 - threshold_shift) / w, -60, 60)))
        keep = hit & (e > 0) & (rng.random(e.shape) < p)
        return np.where(keep, e, 0.0).astype(np.float32)
