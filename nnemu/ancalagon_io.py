"""Readers for Ancalagon compact output (OUTPUT_MODE=compact or both).

Trees (see Ancalagon README, "Compact output, truth, gas pressure and the
gamma library"):
  BgoEvent  one row per simulated event; per-crystal summed edep (unsmeared)
  Truth     one row per simulated event (reaction mode)
  LibTruth  one row per simulated event (gamma-library mode)
  RunInfo   one row per run

Everything here returns raw, unsmeared deposits. The detector response
(resolution, thresholds) is applied later, so a new calibration never needs
new Geant4 runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import awkward as ak
import numpy as np
import uproot

N_CRYSTALS = 30


@dataclass
class CompactRun:
    n_events: int
    crystal_id: ak.Array      # [event][hit] int, 1-30
    edep_mev: ak.Array        # [event][hit] float, summed per crystal
    t_first_ns: ak.Array      # [event][hit]
    dsssd_recoil: np.ndarray  # [event] 0/1
    run_info: dict
    truth: dict | None = None      # reaction mode
    lib_truth: dict | None = None  # gamma-library mode

    # --- standard observables (raw deposits, no response) -------------
    def hit_mask(self) -> np.ndarray:
        return ak.to_numpy(ak.num(self.edep_mev) > 0)

    def singles(self) -> np.ndarray:
        """Highest single-crystal deposit per event (0 for no-hit events)."""
        return ak.to_numpy(ak.fill_none(ak.max(self.edep_mev, axis=1), 0.0))

    def addback(self) -> np.ndarray:
        """Sum over all crystals per event."""
        return ak.to_numpy(ak.sum(self.edep_mev, axis=1))

    def multiplicity(self, threshold_mev: float = 0.0) -> np.ndarray:
        return ak.to_numpy(ak.sum(self.edep_mev > threshold_mev, axis=1))

    def dense_deposits(self) -> np.ndarray:
        """[n_events, 30] float32 matrix of per-crystal deposits."""
        out = np.zeros((self.n_events, N_CRYSTALS), dtype=np.float32)
        counts = ak.to_numpy(ak.num(self.crystal_id))
        rows = np.repeat(np.arange(self.n_events), counts)
        cols = ak.to_numpy(ak.flatten(self.crystal_id)) - 1
        out[rows, cols] = ak.to_numpy(ak.flatten(self.edep_mev))
        return out


def read_compact(path: str, truth: bool = True) -> CompactRun:
    with uproot.open(path) as f:
        ri = {k: v[0] for k, v in f["RunInfo"].arrays(library="np").items()}
        ev = f["BgoEvent"].arrays(["crystalID", "edepMeV", "tFirstNs", "dsssdRecoil"], library="ak")
        n = int(ri["nEvents"])
        if len(ev) != n:
            raise ValueError(f"{path}: BgoEvent has {len(ev)} rows but RunInfo.nEvents={n}")
        tr = lt = None
        if truth and "Truth" in f:
            tr = f["Truth"].arrays(library="ak")
        if truth and "LibTruth" in f:
            lt = f["LibTruth"].arrays(library="np")
    return CompactRun(
        n_events=n,
        crystal_id=ev["crystalID"],
        edep_mev=ev["edepMeV"],
        t_first_ns=ev["tFirstNs"],
        dsssd_recoil=ak.to_numpy(ev["dsssdRecoil"]),
        run_info=ri,
        truth=tr,
        lib_truth=lt,
    )


def read_library_cells(path: str) -> np.ndarray:
    """Cells file used for an Ancalagon --gamma-library run: [n_cells, 3]
    columns (E0_MeV, z_cm, beta)."""
    return np.atleast_2d(np.loadtxt(path, comments="#"))
