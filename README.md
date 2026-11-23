# DRAGON NN emulator

A bin-by-bin neural-network emulator of DRAGON BGO gamma-ray spectra. It covers
arbitrary cascades, beams, H2/He targets and gas pressures, and is trained on
Ancalagon (Geant4) simulations.

The full design is in `PLAN.md`. Briefly:

- Spectra are exact linear mixtures over cascade paths and over vertex z.
- The network therefore only learns S(gamma path, z, beta, response,
  acceptance).
- Branchings, pressure, beam energy and gas enter outside the network, through
  path enumeration and p(z).
- Training targets come from a single-gamma Ancalagon deposit library,
  assembled event by event into cascades.
- Full-cascade Ancalagon runs are the validation gate.

Predecessors, kept as references:

- `~/data/DRAGON_G3_Emulator` (Geant3)
- `~/data/Ancalagon_k39_emulator` (physics response function)

## Status

**Phase 1 (Ancalagon I/O) is done** on Ancalagon branch `nn-emulator-io`,
built in `~/codes/Ancalagon/build-nn`. It adds:

- compact per-event output (`OUTPUT_MODE=compact`), with `BgoEvent` summed per
  crystal plus `Truth`/`LibTruth` and `RunInfo` trees;
- `OUTPUT_FILE`, `GAS_PRESSURE_TORR` and `GAS_TEMPERATURE_K`;
- `--gamma-library` mode (`LIBRARY_CELLS`, `LIBRARY_EVENTS_PER_CELL`);
- `reactions/reaction_writer.py`, a general reaction-file writer for any beam,
  target and branching scheme.

See the Ancalagon README section "Compact output, truth, gas pressure and the
gamma library" for details. Checks done:

- With no new variables set, legacy output is bit-identical to `main`.
- `scripts/regression_k39.py` reproduces the cached k39 spectra above 0.5 MeV,
  with worst chi2/ndf 1.37 over 8 spectra.
- A pressure scan moves the vertex as expected.
- Single gammas cost about 0.05 ms/event and 37 B/event.

**Next: Phase 2.** Build Ancalagon on the cluster (c102), write the condor
wrapper and submit tooling, and run a pilot.

## Layout

| path | contents |
|---|---|
| `nnemu/ancalagon_io.py` | reader for compact output (`read_compact`: singles, addback, multiplicity, dense 30-crystal deposits), and `read_library_cells` |
| `scripts/regression_k39.py` | Phase-1 regression against the cached k39 dataset |
| `responses/` | detector-response files (`--response`), default `s1805.yaml` (to come) |
