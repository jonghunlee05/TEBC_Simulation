# Known limitations

This document is the honest counterpart to `README.md` and
`TEBC_implementation_spec.md`. The README describes the framework as it
is *envisioned*; this file lists the gaps between that vision and the
code that actually runs in `tebc.orchestrator.run_pipeline`. Read both
before drawing conclusions from any number this code prints.

Status legend: 🔴 active blocker for publication-quality numbers · 🟡
significant simplification, defensible if disclosed · 🟢 known small
issue / cosmetic.

Last refreshed alongside the per-interface ERR / Hsueh / Wagner-recession
commit; see `git log` for the history of items previously listed here.

---

## What the orchestrator actually does

`run_pipeline(cfg)` currently executes:

| Scale | What runs | What's documented but skipped |
|-------|-----------|-------------------------------|
| 1 — Atomistic | Database lookup of (E, ν, α, κ, …); isotropic stiffness built from scalar (E, ν) when `dft_outcar` is empty (default). | Real DFT parsing (functions exist, never invoked); DP-GEN active learning; defect-formation chemistry. |
| 2 — MD / phonons | Database lookup of α and κ; `D_O` via Arrhenius from the YSZ entry at the **hot-face** temperature. | Phonopy/QHA, Green-Kubo κ, MSD → D fits — `tebc/scale2_md/` is implemented but not wired into the pipeline. |
| 3 — Mesoscale | Reference-shifted Arrhenius for k_p; Robinson–Smialek k_l with out-of-domain warning; paralinear ODE with optional duty-cycle T schedule; Maxwell–Eucken / Phani–Niyogi homogenization. | Phase-field CMAS attack (`CMASPhaseField.step` raises `NotImplementedError`); CALPHAD equilibrium; TGO microstructure. |
| 4 — Continuum | In-plane CTE bilayer mismatch (Stoney, with general-bilayer Hsueh form available); Hutchinson–Suo steady-state ERR for **per-interface** (EBC channeling, TGO delamination) failure indices; headline = max over interfaces. | A real FEniCSx FEM solver; Lemaitre/Mazars damage and TVH cohesive zones implemented but never invoked from the pipeline. |

So the "four-scale" pipeline is, today, a sequence of database lookups
and analytical closed-form expressions. This is fine as a screening
calculator or surrogate; it is **not** a coupled multi-scale FEA.

---

## 🔴 Active limitations affecting numbers

1. **No experimental validation.** No predicted-vs-measured comparison
   exists for TGO thickness, spallation life, κ(T), or recession. The
   test suite verifies algebraic consistency, not physical accuracy.
   Without at least one validated case the model has no demonstrated
   predictive power.

2. **Sensitivity analysis runs against an analytical surrogate.**
   `tebc.sensitivity.tebc_failure_model` has been re-aligned with the
   pipeline on PBR strain and viscoplastic relaxation, but it still
   omits the temperature schedule, the per-interface ERR split, the
   Wagner recession formula, and the in-plane CTE. The
   `tebc_failure_model_pipeline` companion calls `run_pipeline` for
   each Sobol sample but is **single-threaded** (mutates the global
   `MATERIALS` dict under a lock); a parallel-safe rewrite needs
   `run_pipeline` to accept material overrides directly, which is
   tracked separately.

3. **The Si bond coat is treated purely elastically.** It operates at
   1473–1600 K (Si melts at 1687 K), well into the creep regime. Real
   bond coats accommodate TGO growth strains by viscoplastic flow, and
   `tgo_relaxation_factor` (default 0.07) is a *calibration knob* that
   collapses the ~30 GPa elastic estimate to the 1–3 GPa range that in-
   situ XRD measurements on TBC systems consistently report. The
   number is illustrative; **calibrate per system before publishing**.

4. **`Gamma_interface = 10 J/m²` for SiO₂ TGO is a calibration knob**,
   not a measured per-system value. Mode-I fracture energies of
   ceramic/metal and ceramic/ceramic interfaces in TBC systems span
   5–30 J/m² depending on processing and testing geometry. The 10 J/m²
   default is illustrative; Hutchinson & Suo (1992) is the classic
   mechanics reference but does not provide system-specific numbers.

5. **`run_pipeline` does not carry a temperature gradient through the
   coating** — `T_mean = (T_hot + T_cold)/2` is used for elastic
   softening as a single representative T. A real T/EBC has steep
   through-thickness gradients; a 1-D heat-conduction step is needed
   before Scale 4 stress.

6. **Phonopy QHA mode-following is missing.** Element-wise `log ω`
   differencing across volumes mixes branches at every phonon crossing.
   For monoclinic β-RE silicates this is everywhere.

---

## 🟡 Significant simplifications

7. **Anisotropic CTE handling.** The orchestrator now uses
   `alpha_aniso[0]` (the principal CTE component nominally aligned with
   the in-plane direction). For a textured polycrystal this is
   approximate — the true in-plane CTE depends on the grain orientation
   distribution, which is not modelled. Also, the convention that
   `[0]` corresponds to "the in-plane axis" is undocumented in the DB.

8. **Energy release rate is mode-I only.** No Dundurs α/β parameters,
   no mode-mix ψ. T/EBC interface fractures are mixed-mode and Γ_c
   depends on ψ; this is glossed over.

9. **The `fail_index = max(FI_EBC, FI_TGO)`** treats the two failure
   modes as independent. They aren't: TGO growth raises EBC stress via
   bond-coat constraint, and EBC delamination relieves TGO stress.

10. **Robinson–Smialek runs by default at 200× the calibration v_gas**
    (engine velocities ≫ 4.4 cm/s anchor). A `UserWarning` is now
    emitted when v_gas / P_H2O / T leave 5×/0.2× of the anchor, but the
    correlation itself doesn't extrapolate physically beyond it.

11. **Charged defect formation energies have no Freysoldt correction.**
    `compute_defect_formation_energy` defaults `E_corr = 0.0`. For
    ~100-atom supercells the finite-size electrostatic correction is
    ~1 eV; rankings of point defects (V_O, V_Yb, …) flip without it.

12. **Linear elastic softening with T** in the orchestrator
    (`C_ij_T = C_ijkl · (1 - 0.15·(T-300)/1300)`) is an unsourced
    placeholder and ignores material-specific elastic anomalies.

13. **Two sources of truth for material parameters**:
    `data/materials_db.json` and `tebc.constants.MATERIALS`. A sync
    test now catches drift on shared keys, but the structural fix is
    one loader feeding both. Until then, edits must touch both files.

14. **`phi_TBC = 0.12` and `phi_EBC = 0.03`** are still defaults;
    APS-YSZ porosity is process-dependent (8–25 %). They are now
    `TEBCConfig` knobs.

15. **β-Yb₂Si₂O₇ vs β-Y₂Si₂O₇ oxidation kinetics are nearly identical
    in the database** (k_p ratio ~1.25); experimentally the Yb form is
    meaningfully more oxidation-resistant. Database needs a refresh
    against current literature.

16. **Cyclic damage is not modelled.** The Scale 4 failure index is a
    single-event quasi-static check; no fatigue or ratcheting.
    Duty-cycle T schedule (now supported) only affects TGO growth.

17. **Hsueh's general bilayer formula is implemented**
    (`bilayer_mismatch_stress_hsueh`) but not used by `run_pipeline`,
    which still calls the Stoney-limit `bilayer_mismatch_stress`.

18. **Wagner / paralinear substrate recession** is now used (was
    `k_l · t`); `solve_paralinear` accepts a `PBR` argument with a
    SiO₂/Si default of 2.15. Not pulled from the materials DB
    automatically.

18a. **Units convention for `k_l` is potentially ambiguous.** The
     paralinear ODE `dx/dt = k_p/(2x) − k_l` treats `k_l` as an
     oxide-thickness loss rate (so that `x_ss = k_p/(2k_l)` is the
     oxide steady-state thickness). The Robinson–Smialek calibration
     anchor `RS_K_L_REF = 2e-9 m/s` is, in the Opila/Robinson–Smialek
     literature, conventionally a *substrate (Si) recession rate*.
     Whether the database `k_l` values are stored in oxide-rate or
     substrate-rate units has not been audited against the primary
     references; mixing the two introduces an implicit factor of PBR.
     The Wagner-corrected recession therefore should be treated as
     illustrative until this is resolved.

18b. **Volatilization-dominated regime (oxide consumed) is not
     modelled.** Once volatilization outpaces oxidation, the oxide
     thickness reaches zero and the substrate is directly attacked by
     the gas. The Wagner recession formula
     `(x − x₀ + k_l·t)/PBR` continues to grow at `k_l/PBR` in this
     regime, but physically the rate should switch to direct
     substrate volatilization at `k_l` (no PBR). For realistic T/EBC
     operation the protective oxide stays intact, so this is rarely
     reached, but the orchestrator does not detect or warn about
     entering it.

18c. **`T_schedule` varies temperature only.** Real engine cycles also
     modulate P_H2O, P_O2, and v_gas (lower during cold soak). The
     duty-cycle integrator holds these constant; only T is scheduled.
     A full multi-environment cycle would require per-segment
     (T, P_H2O, P_O2, v_gas) tuples.

18d. **Schedule with cold-T entries triggers R–S out-of-domain
     warnings on every cold-segment evaluation.** The orchestrator
     does not suppress these, so a hot-warm-cold schedule prints one
     warning per cold-T point. Cosmetic.

18e. **`tgo_growth_stress(x_TGO, ...)` ignores `x_TGO`.** The biaxial
     stress is thickness-independent under the elastic treatment;
     the parameter is retained for API stability and to allow future
     thickness-dependent plastic-relaxation models.

---

## 🟢 Smaller residuals

19. **`compute_hcacf` defaults to the biased 1/n estimator** standard
    in Green–Kubo. An `unbiased=True` flag is now available; documented
    but defaults unchanged so historical results are reproducible.

20. **`integrate_tgo_temperature_schedule` averaging** is exact only
    when one cycle ≪ growth timescale. A `cycle_period` argument now
    triggers a `UserWarning` if this assumption breaks down.

21. **CI heavy-test job** runs the conda-forge stack
    (pymatgen / fipy / pycalphad / phonopy / ase) but is marked
    `continue-on-error` because the conda solve is slow and fragile.
    Coverage on the heavy modules therefore exists but is not
    blocking.

22. **`run_pipeline` writes no `sobol_indices.csv` by default** —
    opt-in via `TEBCConfig.write_sobol_csv`. The CLI `__main__` block
    still does not opt in; users running `python -m tebc.orchestrator`
    won't see the CSV.

23. **`extract_born_effective_charges` enforces the acoustic sum rule**
    by subtracting the mean across atoms. Tested.

24. **`parse_elastic_tensor` returns Pa** (was GPa). Unit-conversion
    test exists. Behaviour with newer pymatgen versions that already
    return GPa is documented but not auto-detected.

---

## What this is good for

* Rapid screening of material choices when only relative differences
  matter.
* Sensitivity studies of the analytical surrogate (with caveat #2).
* Algorithm prototyping for the eventual coupled FEA.
* A scaffold for future contributors to fill in real Scale 2 / 3 / 4.

## What this is **not** good for

* Quantitative life prediction of any particular T/EBC system.
* Comparative ranking of two coatings whose differences fall inside
  the simplifications listed above (≈ a factor of 3 in either direction).
* Anything that requires the documented FEA, phase-field, or QHA
  modules to actually have run.
