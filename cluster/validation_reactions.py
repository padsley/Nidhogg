"""Reactions and cascades for the full-cascade validation set ("val_v1").

These are kinematic benchmarks for the factorization gate and end-to-end
tests, not evaluated nuclear data. Resonance energies are representative of
the DRAGON measurements named in PLAN.md. Level schemes are ILLUSTRATIVE:
real low-lying levels, but branchings are chosen only to give a mix of
multiplicities. Charge states are representative; the separator is
auto-retuned for each, and transmission was shown to be q-independent
(Phase 2).
"""
import numpy as np

# name: (beam, target, recoil, charge_state, Er_MeV, illustrative levels {n: Ex}, branches)
REACTIONS = {
    "k39pg": ("39K", "1H", "40Ca", 8, 0.606,
              {1: 3.73669, 2: 3.90438, 3: 6.750},
              [(-1, 40, 0), (-1, 30, 2), (-1, 20, 3), (-1, 10, 1), (3, 100, 2), (2, 80, 0), (2, 20, 1), (1, 100, 0)]),
    "k38pg": ("38K", "1H", "39Ca", 8, 0.679,
              {1: 2.467, 2: 2.796},
              [(-1, 50, 0), (-1, 30, 1), (-1, 20, 2), (2, 100, 0), (1, 100, 0)]),
    "f19pg": ("19F", "1H", "20Ne", 4, 0.323,
              {1: 1.63367, 2: 4.24770, 3: 4.96677},
              [(-1, 30, 0), (-1, 40, 1), (-1, 30, 3), (3, 100, 1), (2, 100, 1), (1, 100, 0)]),
    "ne22pg": ("22Ne", "1H", "23Na", 5, 0.458,
               {1: 0.43999, 2: 2.39100},
               [(-1, 50, 1), (-1, 30, 2), (-1, 20, 0), (2, 70, 1), (2, 30, 0), (1, 100, 0)]),
    "ne22ag": ("22Ne", "4He", "26Mg", 5, 0.83,
               {1: 1.80874, 2: 2.93831},
               [(-1, 50, 0), (-1, 30, 1), (-1, 20, 2), (2, 80, 1), (2, 20, 0), (1, 100, 0)]),
    "o15ag": ("15O", "4He", "19Ne", 4, 0.5036,
              {1: 0.23804, 2: 1.50790},
              [(-1, 80, 0), (-1, 15, 1), (-1, 5, 2), (2, 100, 0), (1, 100, 0)]),
}

PRESSURES = (1.0, 4.0, 10.0)
VERTEX_TARGETS_CM = (-2.5, 2.5)   # aimed-for mean vertex z (cell spans +-5.315)
N_RANDOM_CHAINS = 10


def random_chains(e_budget: float, n: int, seed: int) -> list[list[float]]:
    """n linear chains of multiplicity 1-5 (cycled), gamma energies from a
    uniform ordered-simplex split of e_budget with every gamma >= 0.1 MeV."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        m = 1 + i % 5
        while True:
            cuts = np.sort(rng.uniform(0, e_budget, m - 1))
            g = np.diff(np.r_[0.0, cuts, e_budget])
            if (g >= 0.1).all():
                break
        out.append(sorted(g.tolist(), reverse=True))
    return out
