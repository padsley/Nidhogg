# Plan: bin-by-bin neural-network emulator of DRAGON BGO spectra (Ancalagon, multi-beam, H/He, arbitrary cascades)

## Context

Earlier emulators in `~/data/DRAGON_G3_Emulator` (Geant3) and `~/data/Ancalagon_k39_emulator` had three limits:
- They were trained on one reaction (³⁹K(p,γ)), at a fixed vertex (z = 0), with one Poisson-noisy run per parameter point (8–13k events).
- They used a single observable: the highest-energy-crystal or addback spectrum.
- Direct NN and PCA architectures failed (53–112% mean bin error), mainly because:
  - the targets were noise-dominated,
  - an MSE-type loss was used,
  - and there were too few, too-correlated training points.

  In this setup, PCA and MLP cannot represent a peak that moves continuously.

The goal is an emulator that is non-parametric, i.e. bin-by-bin, with no assumed spectral shape, and that:
- covers many beams, H₂ and He targets, gas pressures and beam energies, which set where the vertex is and therefore the Doppler shift seen by each crystal;
- accepts arbitrary cascade models;
- predicts several observables jointly: E_γ1, the sum spectrum, the multiplicity, the hit pattern / BGO z-centroid, and **individual-crystal spectra**;
- is fast and differentiable enough for likelihood fits and MCMC;
- is validated by closure tests against full Ancalagon simulations, not just by bin-error metrics.

## Key findings that shape the plan

**Ancalagon as it stands** (`~/codes/Ancalagon`):
- The `Bgo` tree has one row per Geant4 *step*, including zero-energy steps. That is why the last campaign used 49 GB.
- **No per-event truth is stored**: no vertex z, true Eγ or directions, recoil momentum, cascade path or DSSSD flag.
- Gas pressure is hard-coded to 5 Torr (`src/TargetChamber.cc:40`) and there is no temperature.
- There is no resolution, threshold or PMT model.
- Every k39 run so far used a fixed vertex. A realistic vertex distribution exists behind the `BKIN` + `RWID` cards (`src/PrimaryGeneratorAction.cc:85-115`), but those runs never used it.
- It runs single-threaded and always writes `dragon_hits.root` into the current directory.
- Lifetimes are parsed but ignored, so every γ is emitted at the vertex.
- Angular distributions are isotropic.

**Cluster:**
- Submit host: `c102.cyclotron.tamu.edu`, using `ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/cluster`.
- Data goes in `/data/pagroup/pa02/padsley/…`. The 25 GB home quota is almost full.
- Geant4 11.3 and ROOT are already installed under `/home/pagroup/padsley/codes/`; this is the same Geant4 version as the local Ancalagon build.
- Template to copy: the X17 cosmics `.sub` and wrapper at `/data/pagroup/pa02/X17_cosmics/`.
  - `should_transfer_files = NO`; both `FileSystemDomain`s; one condor log per cluster; `max_materialize`; ≤100 job starts per 5–10 min.
- **Ancalagon has never been built on the cluster.**
- Never write large temporary files to `/tmp`.
- **Local disk:** 12 GB free (98% full), no GPU, 16 cores, 31 GB RAM.

**Literature:**
- **Current DRAGON practice:** branch-template fits plus profile scans over discrete simulations. Examples are Williams 2022 (TFractionFitter on E_γ1) and Christian 2018 (Poisson-likelihood fit of the BGO z-distribution with a measured Fermi-function threshold and gas-density profile). The emulator replaces those discrete grids.
- **Summing detectors** (SuN, Dombos 2016; DANCE/DICEBOX, Valenta 2015):
  - They fit the sum, segment and multiplicity spectra **simultaneously**. Segment and multiplicity spectra break degeneracies between cascades that the sum spectrum alone cannot.
  - SuN went to more segments for exactly this reason.
  - This is what supports the request for per-crystal spectra.
- **Total-absorption spectroscopy (TAS) response construction** (Cano-Ott 1999; Tain & Cano-Ott 2007):
  - Level responses are built recursively from single-γ responses.
  - This is only valid at the histogram level if the showers are independent.
  - Assembling at the **event** level, by adding crystal deposit vectors before applying thresholds and taking the maximum, keeps summing in the same crystal and the non-linearities correct.
- **Machine-learning surrogates:**
  - Poisson or multinomial negative log-likelihood (NLL) with a log-link output: its minimiser is the expectation, so noisy MC targets are fine.
  - Deep ensembles for emulator uncertainty, with that variance propagated into the fit likelihood.
  - Per-bin surrogates in the style of Professor and MCNNTUNES.
  - The closest gamma-spectrum analogue is the MC-spectrum emulator with Bayesian inference in Breitenmoser 2023.
  - Calorimeter generative models (CaloFlow and similar) are only relevant for validation practice.
- **Design of experiments:** Sobol / Latin hypercube sampling plus active learning; group-held-out validation; simulation-based calibration and closure tests.

## Core idea: three exact linear decompositions

The observed (expected) spectrum for a reaction is:

```
S_obs(bins) = Σ_paths P(path | level scheme, branchings)
            × ∫ dz p(z | E_beam, E_r, Γ, pressure, gas, density profile)
            × S(bins | γ-path {E_γ1..E_γn}, z, β; detector response, recoil acceptance)
```

Events are independent, so the mixtures over **cascade paths** and over **vertex position** are exact linear sums, even though thresholds, max-crystal selection and summing in the same crystal are non-linear *within* a path. As a result:
- **Branching ratios and level schemes never need to be learned.** Use `parse_reaction.ReactionFile.cascades()` from the k39 repo to enumerate the paths, then take the weighted sum.
- **Pressure, beam energy, resonance energy and width, density profile and straggling only enter through p(z).** p(z) is computed outside the network from stopping powers, as Christian 2018 and Ancalagon's `GasStoppingPower` do. This separates the vertex/pressure confounder from the cascade physics by construction.
- **Beam and target species enter the network only through:**
  - β of the recoil at emission;
  - the recoil-acceptance parameter (p_recoil versus the separator acceptance);
  - p(z).

  The BGO response to a γ does not depend on which nucleus emitted it. This is tested explicitly: two beam/target pairs with the same β must give identical S.
- **The network learns only S(path, z, β, response, acceptance)**, with roughly 6–10 continuous inputs plus a set of 1–6 γ energies. That is a far smaller problem than "level scheme × beam × target × pressure".

## Data engine: per-γ event library plus event-level assembly, validated against full cascades

Simulating every (cascade × z × β) combination with full Ancalagon at high statistics is wasteful and noisy. Instead:

1. **Single-γ deposit library (cluster).** Ancalagon emits one γ per event:
   - energy E₀ in the recoil frame, isotropic, at a fixed vertex z, boosted by a fixed β;
   - the grid is E₀ in 20 keV steps up to about 12–15 MeV × z from −8 to +8 cm in 1 cm steps (covering the density ramps in the collars) × about 6 β values that span the beams of interest;
   - about 2×10⁴ events per cell.

   For each event, store:
   - the sparse 30-crystal deposit vector (crystal ID and energy for crystals with energy > 0), with no smearing;
   - the emission direction in the recoil frame (θ*, φ*), which is used later for the recoil kick.

   Rough budget:
   - about 6×10⁴ cells, about 1.2×10⁹ events, about 500 CPU-hours (about 1.5 ms per event for one γ, from the measured ~3 ms per event for an 8.9 MeV cascade);
   - about 20 GB on `/data/pagroup/pa02`;
   - roughly a day of wall time under the c102 ramp limits.
2. **Assembly (numpy/numba, on the cluster).** For a path {E_γi} at (z, β):
   - Draw one library event per γ from its E-cell. Scale deposits by E/E_cell to cover off-grid energies (≤10 keV); this error is far below BGO and Doppler widths and is checked in validation.
   - Add the vectors crystal by crystal.
   - Apply the **detector response**: per-crystal Gaussian resolution σ(E) from the DRAGON BGO calibration work, a per-crystal Fermi-function hardware threshold (Christian 2018 form), and a global resolution scale and threshold shift as nuisance inputs.
   - Compute **recoil acceptance** by summing the γ momenta to get the recoil kick angle, then weighting each event by a transmission map from dedicated Ancalagon recoil runs.
   - Fill every observable head (below).

   Any number of events can be assembled cheaply, so training targets are effectively noise-free.
3. **Full-cascade Ancalagon validation set (cluster).**
   - About 6 real reactions, e.g. ³⁹K(p,γ), ³⁸K(p,γ), ²²Ne(p,γ), an (α,γ) case on He; the user confirms the list.
   - About 30 cascades each, multiplicity 1–5, including real multi-branch schemes.
   - Realistic `BKIN` + `RWID` vertex distributions, 2–3 pressures, both gases, 2×10⁵–10⁶ events each.
   - About 400 runs, about 100 CPU-hours.
   - This is the ground truth for the **factorization gate** (assembly versus full cascades) and for the final end-to-end tests. If the gate fails for some observable (for example, recoil gating for light recoils), train that part directly on full-cascade runs instead.

This keeps the user's goal of high-statistics runs over many beams, targets and cascade models. The library makes the cascade dimension cheap, and full Ancalagon remains the referee.

## Required Ancalagon changes (in `~/codes/Ancalagon`)

1. **Compact per-event output.** Replace or supplement the per-step `Bgo` ntuple with one row per event:
   - `eventID`, `nCrystals`, a vector of `crystalID`, a vector of `edep` (summed per crystal), first-hit time per crystal, and the DSSSD-hit flag;
   - drop zero-energy steps.

   Files: `src/RunAction.cc`, `src/EventAction.cc`, `src/BgoSD.cc`. Write `nEventsSimulated` to the file so the count no longer depends on parsing `run.log`.
2. **Truth ntuple.** Fill it from `PrimaryGeneratorAction` (`src/PrimaryGeneratorAction.cc:114`, which is where vertex z is computed) via a per-event user-info struct:
   - vertex x/y/z, beam KE at the reaction, E_cm;
   - recoil p and β, cascade path ID;
   - each γ's E and direction in both the lab and the CM frame.
3. **Gas configuration from the environment or a card.** Add `GAS_PRESSURE_TORR` and `GAS_TEMPERATURE_K`. Density must scale as P/T for the main target and the collar/pumping fractions (`src/TargetChamber.cc:110-114,179,242,462-466`).
4. **Library generator mode.** Set by `GUN_MODE=single_gamma` with `GAMMA_E0_MEV`, `VERTEX_Z_CM`, `RECOIL_BETA` (or a list or range), recording CM emission directions. Keep the separator off or fast in this mode: BGO only, no recoil tracking.
5. **Output name and seed hygiene.** Add `OUTPUT_FILE`, and require `RANDOM_SEED` in the batch wrapper. Set `RTUN` explicitly for each reaction so jobs skip the 300-event auto-retune.
6. **Generator scripts.** Generalise `reactions/generate_reactions.py` beyond k39 and beyond 100% linear chains: any BEAM/TARG, branching cascades, `BKIN`/`RWID`.
7. **Regression check.** Re-run 10 k39 stems and confirm the new compact output reproduces the cached `Y_singles`/`Y_addback` in `~/data/Ancalagon_k39_emulator/dataset.npz`. Confirm the DSSSD fraction is about 1 for post-21-Sep optics (stale-template lesson).

## Cluster pipeline (new `cluster/` directory in the new repo)

- Build Ancalagon on c102 against `/home/pagroup/padsley/codes/geant4_11.3_install`, not on c104/c109, which have old CPUs.
- **Wrapper `run_ancalagon_job.sh`** (pattern copied from X17 `run_cosmic_condor.sh`):
  - export the G4 data variables;
  - create a per-job directory under `/data/pagroup/pa02/padsley/dragon_nn/…`;
  - set `REACTION_INPUT`/`GUN_MODE`, `RANDOM_SEED = f(Cluster, Process)` and `TRAJ_QUIET=1`;
  - put the stdout log on `/data`.
- **Submit files** use `queue … from jobs.txt`, `max_materialize`, a ramp under the shared `c102_ramp.pause` convention, and one log per cluster.
- **On-cluster reduction:**
  - an integrity-check script;
  - the library packer, which writes per-(E, z, β) sparse arrays to `.npz`/parquet;
  - the assembler, which writes training-set shards (float16).

  Only the training shards (a few GB) and validation spectra are pulled back with `scp`. Raw ROOT stays on `/data`.

## Network design

- **Inputs:**
  - the γ set {E_γi} (1–6 tokens);
  - the conditions (z, β, resolution scale, threshold shift, acceptance parameter);
  - later, a per-γ emission offset (β·c·τ) so finite lifetimes can be handled. Ancalagon ignores lifetimes; the library supports this with a per-γ z.
- **Output heads:** all are probabilities per emitted event, as log-softmax distributions with an explicit "nothing above threshold" bin so normalisation and efficiency come out together.
  1. E_γ1, the highest single-crystal energy: 0–15 MeV, about 30 keV bins.
  2. Sum energy of crystals above threshold (and addback).
  3. **Individual-crystal spectra, 30 × about 250 bins.** A crystal-conditioned decoder takes as inputs the crystal ID embedding and geometric features (polar angle and distance relative to the vertex at this z). Statistics are shared across crystals, and the Doppler/angle geometry is explicit.
  4. Crystal multiplicity above threshold.
  5. Crystal hit pattern of the E_γ1 crystal, which gives the BGO z-centroid used by DRAGON.
  6. Optional, later: a coarse E_γ1 × E_γ2 matrix.
- **Architecture:**
  - A Set Transformer encoder over the γ tokens with FiLM conditioning on (z, β, …).
  - Decoders are **coordinate/query-based**: each output bin (and crystal) is a query that cross-attends to the γ tokens, with *relative-energy features* (E_bin − D·E_γi), (E_bin − ΣE_γ), (E_bin − (E_γi + E_γj)), (E_bin − E_γi + 511/1022 keV).

    This is still bin-by-bin and non-parametric, but the network is told where peaks, sum peaks and escapes *can* be. That fixes the shifting-peak failure of the old PCA and absolute-energy implicit networks.
  - **Baseline:** a wide plain MLP that maps inputs to all bins, which is the "large network" in its simplest form. It is trained identically for comparison.
- **Loss:** multinomial (cross-entropy) NLL against the assembled target distributions, weighted by event count, i.e. the Poisson deviance. Training uses AdamW with a cosine schedule.
- **Uncertainty:** an ensemble of 5 networks. The ensemble variance is added to the fit likelihood.
- **Training set:**
  - About 3×10⁵ (path, z, β, response) samples.
  - Paths come from a Sobol design over multiplicity 1–6 and γ energies under an energy budget of ≤ about 15 MeV, densified below 3 MeV and near 511 keV multiples.
  - Each sample is assembled from 2×10⁵–10⁶ events.
  - This is followed by active-learning rounds that add samples where ensemble disagreement is highest.
- **Hardware:** there is no local GPU, so ask about GPU access (see the questions). The fallback is CPU training on 16 cores or a cluster node.

## Using the emulator

New fitter, generalising `fit_data.EmulatorFitter` from `~/data/DRAGON_G3_Emulator/fit_data.py`:
- It takes a level scheme with free branchings (Dirichlet or simplex parametrisation), E_r, Γ, pressure and gas.
- It computes p(z) on about 15 z nodes.
- It enumerates the paths.
- It runs a batched emulator call over paths × z nodes.
- It applies a joint Poisson likelihood over the chosen heads, plus an emulator-variance term.
- It uses gradient-based optimisation and MCMC (emcee or NUTS via torch autograd).
- Branchings and efficiency then come out of one model, replacing template fits and profile-likelihood grids.

## Confounders handled explicitly

| Confounder | How it is handled |
|---|---|
| E_beam/E_r, pressure, Γ, straggling, density ramps → vertex z | p(z) is computed outside the network, and the mixture over z is exact; z grid covers ±8 cm. |
| z ↔ Doppler shift per crystal ↔ level energies (both move peaks) | **Per-crystal heads plus hit pattern**: a Doppler or z shift has crystal-angle-dependent sign and size, while a level-energy shift is the same for every crystal. Quantified in closure tests. |
| β (from E_beam) coupled to z | β and z are separate network inputs; β comes from kinematics in the fitter. |
| Branchings ↔ multiplicity-dependent threshold efficiency | Normalised heads include the "below threshold" bin; the multiplicity head is fitted jointly. |
| γ directions ↔ recoil acceptance (DSSSD gate) | Event-level kick computed from CM directions plus a transmission map; gated and ungated heads. |
| Resolution and threshold calibration | Nuisance inputs (global scale and shift) on top of the per-crystal calibration. |
| Lifetimes (ignored by Ancalagon) | Documented limitation; per-γ emission offset is a planned extension. |

## Validation (gates in order)

1. **Ancalagon regression.** The new output reproduces the cached k39 spectra; the DSSSD fraction is about 1.
2. **Factorization gate.** Compare assembly with full Ancalagon cascades on every head, including per-crystal: Poisson deviance/ndf ≈ 1 at 10⁶ events, and scan pulls by bin. Also check the E-cell scaling interpolation. *Do not train until this passes.*
3. **Emulator versus assembly.** Random and **group-held-out** tests: an unseen β value, an unseen z value, multiplicity 6, an energy band. Report deviance/ndf at 10⁴, 10⁵ and 10⁶ counts; it must not grow with statistics. Do not use median relative error (a lesson from the old project).
4. **End-to-end versus full Ancalagon** for real reactions with realistic p(z), both gases and two pressures.
5. **Closure tests.** Pseudo-data from **full Ancalagon**, not the emulator, with known branchings and E_r. Fit with the emulator and check bias, pull widths, bias versus statistics, and that no parameter is pinned at a bound.
   - Repeat using the heads incrementally: E_γ1 only → +sum → +multiplicity → +hit pattern → +per-crystal.
   - This quantifies how much the individual-crystal spectra help resolve z/Doppler/branching degeneracies (also via the Fisher information from emulator gradients).
6. **Real-data benchmark.** S1805 ³⁹K(p,γ) at 337/606/666 keV, using the existing fits and calibration in `~/data/DRAGON-BGO-Calibration` and its project memory.

## Phases

0. **Decisions** (questions below), and set up a new repo.
1. Ancalagon changes 1–7 plus the local regression test.
2. Cluster build, wrapper, submit and ramp tooling; a 50-job pilot to measure the real cost per event and the bytes per event.
3. Library campaign and full-cascade validation campaign; on-cluster packing.
4. Assembler plus detector-response module; **factorization gate**.
5. Training-set generation, baseline MLP, the main model, the ensemble, and active learning.
6. Fitter plus closure tests plus the observable-ablation study; real-data benchmark.

## Reuse

- `~/data/Ancalagon_k39_emulator`:
  - `extract_spectrum.py`: uproot reading, `reduceat` per-crystal sums, `n_total`;
  - `parse_reaction.py`: `ReactionFile.cascades()`;
  - `gamma_physics.py`: Doppler, Compton edge and escape helpers for relative-energy features and checks;
  - `response_function.py`: a physics baseline for comparison.
- `~/codes/Ancalagon/reactions/generate_reactions.py` and `build_grid.py`: the RECL trick and Sobol simplex sampling.
- `~/data/DRAGON_G3_Emulator/fit_data.py`: fitter structure, vectorised batch likelihood, emcee.
- `/data/pagroup/pa02/X17_cosmics/` on the cluster: submit and wrapper templates; integrity-check plus hadd pattern.

## Decisions (user, 2026-11-23)
- **Data engine:** library plus assembly, with full-cascade runs as the validation gate.
- **Scope:** a broad DRAGON set from the start.
  - Recoil β grid: 0.010–0.050 in steps of 0.005 (9 values). This covers roughly 0.15–1 MeV/u beams on H₂ (β_recoil ≈ 0.96·β_beam) and He (≈ 0.8·β_beam).
  - Proposed validation reactions:
    - ³⁹K(p,γ)⁴⁰Ca (S1805, real data)
    - ³⁸K(p,γ)³⁹Ca (Christian 2018, z-fit benchmark)
    - ¹⁹F(p,γ)²⁰Ne (Williams 2022; light recoil, so a recoil-gating stress test)
    - ²²Ne(p,γ)²³Na
    - ²²Ne(α,γ)²⁶Mg (He)
    - ¹⁵O(α,γ)¹⁹Ne (He; `reactions/o15ag_19ne.reaction` already exists)

    The user may substitute others.
- **Hardware:** CPU only.
  - Keep the main model about 2–5 M parameters.
  - Use **query-sampled training**: each minibatch evaluates a random subset of (bin, crystal) queries, which the coordinate decoder allows. This bounds memory to the ~7 GB available.
  - Stream float16 training shards via memmap; keep them ≤ 6 GB locally because of the 12 GB free disk.
  - Train the 5 ensemble members sequentially on 16 local cores.
  - Option: whole-node condor jobs (`request_cpus = 24`) on the rack nodes, if the c102 policy allows.
  - The plain-MLP baseline is kept small for the same reason.
- **Repo:** new `~/data/DRAGON_NN_Emulator`, a git repo. The G3 and k39 directories stay as legacy references.

## Remaining open items (resolve in Phase 0)
- ~~Pressure and temperature~~ **Decided (2026-11-23):** pressure 1–10 Torr for both H₂ and He; temperature fixed at 300 K for now.
  - Ancalagon takes `GAS_PRESSURE_TORR`, with `GAS_TEMPERATURE_K` defaulting to 300.
  - Full-cascade validation runs use 1, 4 and 10 Torr.
  - Pressure enters only through p(z), and through recoil straggling in the acceptance map.
- **Decided — detector response:** the default is S1805. Per-crystal resolution and threshold are only known once they are measured for an experiment, so the response must be selectable:
  - A versioned response file such as `responses/s1805.yaml` holds, for each crystal, the σ(E) parameters, the Fermi-threshold E₀ and width, and a gain/scale. It also holds global settings: addback and sum definitions, and the analysis threshold on E_γ1.
  - `s1805.yaml` is the default, built from the S1805 in-situ ⁴⁰K/²⁰⁸Tl calibration in `~/data/DRAGON-BGO-Calibration`.
  - It is selected with `--response <file>` in the assembler, the training scripts and the fitter.
  - Each trained model stores the hash and name of its response file. The fitter refuses to run if the model's response does not match the one requested, unless a flag overrides this.
  - The library stores **unsmeared** deposits, so a new response never needs new Geant4 runs.
  - **Per-crystal heads:** smearing and the threshold are linear operations within a single crystal. So the network emulates *raw* per-crystal deposit spectra, and the response is applied analytically at fit time: a convolution matrix times the threshold-efficiency curve. A new calibration needs no retraining for these heads.
  - **Highest-energy hit, sum, multiplicity and hit-pattern heads:** these depend on thresholds across crystals non-linearly. A new response file therefore means re-assembling the training shards (cheap) and fine-tuning the model from the S1805 weights (CPU hours, not days). The global resolution-scale and threshold-shift nuisance inputs absorb small differences without fine-tuning.
- **Decided:** the γ energy ceiling for the library is 15 MeV.
