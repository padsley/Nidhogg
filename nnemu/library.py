"""Access to a packed single-gamma deposit library (cluster/pack_library.py).

A library is a directory of packed parts (one per Ancalagon --gamma-library
job). Each part holds whole cells (E0, z, beta). Cells across all parts are
indexed by their (E0, z, beta) values. Arrays are memory-mapped, so a 50 GB
library costs no RAM until events are drawn from it.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Part:
    path: str
    cells: np.ndarray        # (n_cells, 3)
    events_per_cell: int
    hit_offset: np.ndarray   # (n_events+1,) uint32, memmap
    crystal: np.ndarray      # (n_hits,) uint8, memmap
    edep: np.ndarray         # (n_hits,) float32, memmap
    cos_theta: np.ndarray    # (n_events,) float16, memmap
    phi: np.ndarray          # (n_events,) float16, memmap


class GammaLibrary:
    def __init__(self, packed_dir: str):
        self.parts: list[Part] = []
        self.index: dict[tuple[float, float, float], tuple[int, int]] = {}
        for name in sorted(os.listdir(packed_dir)):
            d = os.path.join(packed_dir, name)
            if not os.path.isfile(os.path.join(d, "DONE")):
                continue
            meta = json.load(open(os.path.join(d, "meta.json")))
            ld = lambda f: np.load(os.path.join(d, f), mmap_mode="r")  # noqa: E731
            p = Part(d, np.load(os.path.join(d, "cells.npy")), meta["events_per_cell"],
                     ld("hit_offset.npy"), ld("crystal.npy"), ld("edep.npy"),
                     ld("cos_theta.npy"), ld("phi.npy"))
            ip = len(self.parts)
            self.parts.append(p)
            for ic, (e, z, b) in enumerate(p.cells):
                self.index[self._key(e, z, b)] = (ip, ic)
        if not self.parts:
            raise FileNotFoundError(f"no packed parts in {packed_dir}")
        keys = np.array(list(self.index.keys()))
        self.e_values = np.unique(keys[:, 0])
        self.z_values = np.unique(keys[:, 1])
        self.beta_values = np.unique(keys[:, 2])

    @staticmethod
    def _key(e, z, b):
        return (round(float(e), 4), round(float(z), 3), round(float(b), 5))

    def has(self, e, z, b) -> bool:
        return self._key(e, z, b) in self.index

    def energies_at(self, z, b) -> np.ndarray:
        """Grid energies available at a given (z, beta) plane."""
        zk, bk = round(float(z), 3), round(float(b), 5)
        return np.array(sorted(k[0] for k in self.index if k[1] == zk and k[2] == bk))

    def draw(self, e, z, b, n: int, rng: np.random.Generator):
        """n random events (with replacement) from cell (e, z, b). Returns
        (event_of_hit, crystal, edep, cos_theta, phi): hits are labelled by
        draw number 0..n-1."""
        ip, ic = self.index[self._key(e, z, b)]
        p = self.parts[ip]
        ev = ic * p.events_per_cell + rng.integers(0, p.events_per_cell, n)
        lo = p.hit_offset[ev].astype(np.int64)
        hi = p.hit_offset[ev + 1].astype(np.int64)
        cnt = hi - lo
        draw_of_hit = np.repeat(np.arange(n), cnt)
        # hit indices: lo[j] + (0..cnt[j]-1)
        starts = np.repeat(lo - np.r_[0, np.cumsum(cnt)[:-1]], cnt)
        hit_idx = starts + np.arange(cnt.sum())
        return (draw_of_hit, np.asarray(p.crystal[hit_idx]), np.asarray(p.edep[hit_idx]),
                np.asarray(p.cos_theta[ev], dtype=np.float32), np.asarray(p.phi[ev], dtype=np.float32))
