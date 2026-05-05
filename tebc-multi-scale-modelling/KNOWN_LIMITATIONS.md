# Known limitations

This document is the honest counterpart to `README.md` and
`TEBC_implementation_spec.md`. The README describes the framework as it
is *envisioned*; this file lists the gaps between that vision and the
code that actually runs in `tebc.orchestrator.run_pipeline`. Read both
before drawing conclusions from any number this code prints.

Status legend: 🔴 active blocker for publication-quality numbers · 🟡
significant simplification, defensible if disclosed · 🟢 known small
issue / cosmetic.

---

## What the orchestrator actually does

`run_pipeline(cfg)` currently executes:

| Scale | What runs | What's documented but skipped |
|-------|-----------|-------------------------------|
| 1 — Atomistic | Database lookup of (E, ν, α, κ, …); isotropic stiffness built from scalar (E, ν) when `dft_outcar` is empty (default). | Real DFT parsing (functions exist, never invoked); DP-GEN active learning; defect-formation chemistry. |
| 2 — MD / phonons | Database lookup of α(T_mean) and κ; D_O via Arrhenius from the YSZ database entry. | Phonopy/QHA, Green-Kubo κ, MSD → D fits — entire `tebc/scale2_md/` package is implemented but not wired into the pipeline. |
| 3 — Mesoscale | Reference-shifted Arrhenius for k_p, Robinson–Smialek k_l, paralinear ODE for TGO thickness; analytical Maxwell–Eucken / Phani–Niyogi homogenization. | Phase-field CMAS attack (`CMASPhaseField.step` raises `NotImplementedError`); CALPHAD equilibrium; TGO microstructure. |
| 4 — Continuum | Stoney-limit bilayer mismatch stress and Hutchinson–Suo steady-state ERR for two interfaces (EBC channeling, TGO delamination); failure index = max over interfaces of G/Γ. | FEniCSx FEM (`fenics_thermoelastic_setup` returns a *string template*, not a runnable solver); Lemaitre/Mazars damage are implemented but never called. |

So the "four-scale" pipeline is, today, a sequence of database lookups and
analytical closed-form expressions. This is fine as a screening calculator
or surrogate; it is **not** a coupled multi-scale FEA.

---

## 🔴 Active limitations affecting numbers

1. **No experimental validation.** No predicted-vs-measured comparison
   exists for TGO thickness, spallation life, κ(T), or recession. The
   test suite verifies algebraic consistency, not physical accuracy.
   Without at least one validated case the model has no demonstrated
   predictive power.

2. **Sensitivity analysis runs against an analytical surrogate**, not
   the orchestrator. `tebc.sensitivity.tebc_failure_model` is a
   standalone Evans–Hutchinson formula. After every pipeline fix the
   surrogate diverges further from `run_pipeline`. `results/sobol_indices.csv`
   characterises the surrogate, not the code.

3. **TGO is integrated at constant T_hot for the entire `n_cycles` of
   service**, ignoring time spent at intermediate temperatures during
   ramp / dwell / cool. With Ea ≈ 70–100 kJ/mol the rate at 1000 K is
   ~10³× lower than at 1600 K, so the constant-T integral overestimates
   total TGO growth by a large factor. A duty-cycle-weighted T schedule
   is the minimum honest fix.

4. **The Si bond coat is treated purely elastically.** It operates at
   1473–1600 K (Si melts at 1687 K), well into the creep regime. Real
   bond coats accommodate TGO growth strains by viscoplastic flow, which
   is why measured TGO stresses are 1–3 GPa even though the elastic
   PBR-strain calculation predicts ~30 GPa. No plasticity model is
   currently active.

5. **MSD does not subtract centre-of-mass drift.** For oxygen-conduction
   calculations in YSZ this corrupts D_O when the simulation cell has
   any net translation. Standard MD analysis subtracts framework COM
   before differencing.

6. **`compute_msd` is O(N²) in lag.** Current implementation walks every
   lag explicitly; for 10⁵–10⁶-frame trajectories this is unusable.
   FFT-based autocorrelation would be O(N log N).

7. **Phonopy QHA mode-following is missing.** Element-wise `log ω`
   differencing across volumes mixes branches at every phonon crossing.
   Grüneisen γ values for low-symmetry crystals like β-RE silicates
   will be unreliable.

---

## 🟡 Significant simplifications

8. **Anisotropic CTE handling.** The orchestrator now uses
   `alpha_aniso[0]` (the principal CTE component nominally aligned with
   the in-plane direction) instead of the scalar mean. For a textured
   polycrystal this is approximate — the true in-plane CTE depends on
   the grain orientation distribution, which is not modelled.

9. **Energy release rate is mode-I only.** No Dundurs α/β parameters,
   no mode-mix ψ. T/EBC interface fractures are mixed-mode and Γ_c
   depends on ψ; this is glossed over.

10. **Stoney bilayer assumption.** `bilayer_mismatch_stress` ignores
    the h_film/h_substrate ratio. For TBC+EBC ≈ 350 µm on a 5 mm CMC
    the ratio is 0.07 — at the edge of Stoney's validity (≤ 0.1).
    Hsueh's general bilayer formula should be substituted when the
    substrate is thinner.

11. **PBR growth strain uses (PBR−1)/3** (small-strain limit). For
    SiO₂ on Si with PBR = 2.15 the correct linear strain is
    PBR^(1/3) − 1 ≈ 0.29 vs the implemented 0.38. The elastic
    prediction is in any case 10× larger than measured TGO stresses
    because of bond-coat plastic relaxation (limitation #4).

12. **Charged defect formation energies have no Freysoldt correction.**
    `compute_defect_formation_energy` defaults `E_corr = 0.0`. For
    ~100-atom supercells the finite-size electrostatic correction is
    ~1 eV; rankings of point defects (V_O, V_Yb, …) flip without it.

13. **Linear elastic softening with T** in the orchestrator
    (`C_ij_T = C_ijkl · (1 - 0.15·(T-300)/1300)`) is an unsourced
    placeholder and ignores material-specific elastic anomalies.

14. **Two sources of truth for material parameters**:
    `data/materials_db.json` and `tebc.constants.MATERIALS`. They are
    consistent today; the structure invites future drift.

15. **`phi_TBC = 0.12` and `phi_EBC = 0.03`** are hardcoded; APS-YSZ
    porosity is process-dependent (8–25 %). They should become
    `TEBCConfig` knobs.

16. **β-Yb₂Si₂O₇ vs β-Y₂Si₂O₇ oxidation kinetics are nearly identical**
    in the database; experimentally the Yb form is meaningfully more
    oxidation-resistant. Database values need refresh.

17. **Cyclic damage is not modelled.** `n_cycles` enters only through
    the TGO integration window (and even there incorrectly — see #3).
    The Scale 4 failure index is a single-event quasi-static check; no
    fatigue or ratcheting accumulation.

18. **No oxygen partial pressure (P_O₂) parameter.** `TEBCConfig`
    exposes P_H2O and P_tot; P_O₂ is not configurable.

---

## 🟢 Smaller residuals

19. **`mazars_damage` saturates at D = 0.999** rather than D → 1; leaves
    a residual stiffness 0.001·E forever. Standard practice switches to
    element deletion / cohesive at D → 1.

20. **`benzeggagh_kenane_toughness` hardcodes G_IIc = 1.5·G_Ic.** Should
    be a parameter.

21. **`cahill_pohl_kappa_min` uses a single Debye temperature** for all
    branches; the original CWP form has Θ_i per branch.

22. **`phani_niyogi_modulus` φ_c = 0.6 default** is too high for typical
    APS coatings (closer to 0.4–0.5).

23. **`run_pipeline` writes `sobol_indices.csv` unconditionally** to
    `results/`. No timestamp or provenance hash; race condition in
    parallel runs.

24. **CI does not install the heavy modules** (pymatgen, fipy,
    pycalphad, dolfinx). Coverage of every Scale 1, the phase-field /
    CALPHAD parts of Scale 3, and Scale 4 FEA remains zero in CI.

25. **Robinson–Smialek extrapolation is unguarded outside its
    calibration window.** Default `v_gas = 10 m/s` is ~200× the
    Opila/Hann anchor of 4.4 cm/s; the v^0.5 dependence is empirical
    and the boundary-layer regime changes long before this velocity.
    The orchestrator only warns in a comment.

26. **Sobol/Morris bounds in `DEFAULT_TEBC_PROBLEM`** are unsourced.
    Are these literature ranges, processing windows, or judgment?
    Reviewer cannot tell.

---

## What this is good for

* Rapid screening of material choices when only relative differences
  matter.
* Sensitivity studies *of the analytical surrogate* (with caveat #2).
* Algorithm prototyping for the eventual coupled FEA.
* A scaffold for future contributors to fill in real Scale 2 / 3 / 4.

## What this is **not** good for

* Quantitative life prediction of any particular T/EBC system.
* Comparative ranking of two coatings whose differences fall inside
  the simplifications listed above (≈ a factor of 3 in either direction).
* Anything that requires the documented FEA, phase-field, or QHA
  modules to actually have run.
