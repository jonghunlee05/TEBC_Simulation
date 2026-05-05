"""
Main pipeline orchestrator.

Runs the full 4-scale TEBC simulation in sequence:
  Scale 1 → Scale 2 → Scale 3 → Scale 4 → Sensitivity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("tebc")


@dataclass
class TEBCConfig:
    """Top-level simulation configuration."""
    material_EBC:   str = "beta_Yb2Si2O7"
    material_TBC:   str = "7YSZ"
    material_bond:  str = "Si_bondcoat"
    material_sub:   str = "SiC_SiC_CMC"

    h_TBC:    float = 200e-6
    h_EBC:    float = 150e-6
    h_bond:   float = 100e-6
    h_sub:    float = 5e-3

    # Layer porosities — kept distinct because APS YSZ topcoats are far more
    # porous than dense EBC silicates. The earlier code used a single 0.12
    # value for both, which derated the dense EBC modulus by ~50 %.
    phi_TBC:  float = 0.12     # APS-YSZ typical (8–25 %)
    phi_EBC:  float = 0.03     # dense β-RE silicate typical (≤ 5 %)

    # Operating conditions.
    # Defaults reflect a representative aero-engine combustor environment
    # within the calibration window of the Robinson–Smialek correlation
    # (Opila & Hann 1997; calibration anchors ≈ 1316 °C, P_H2O = 0.1 atm,
    # P_tot = 1 atm, v_gas ≈ 4.4 cm/s).  Extrapolating far beyond these
    # anchors (e.g. v = 100 m/s, P_tot = 10 atm) gives empirically
    # nonsensical recession rates — keep these knobs near the anchors and
    # treat large excursions as out-of-domain.
    T_hot:    float = 1600.0   # K
    T_cold:   float = 300.0    # K
    T_dep:    float = 1473.0   # K
    P_H2O:    float = 1.0e4    # Pa  (~ 0.1 atm partial pressure)
    P_O2:     float = 5.0e3    # Pa  (~ 0.05 atm; controls dry-Si oxidation)
    P_tot:    float = 1.0e5    # Pa  (~ 1 atm)
    v_gas:    float = 10.0     # m/s
    n_cycles: int   = 600

    # Calibrated knock-down on the *elastic* TGO growth stress to
    # represent viscoplastic relaxation in the Si bond coat. The elastic
    # PBR estimate predicts ~30 GPa; measured TGOs sit at 1–3 GPa, hence
    # a default of 0.05–0.10. Set to 1.0 to retain the elastic prediction.
    # See KNOWN_LIMITATIONS.md #4.
    tgo_relaxation_factor: float = 0.07

    # Optional duty-cycle schedule for cyclic service. If None, the TGO
    # integration is isothermal at T_hot (overestimates total growth —
    # see KNOWN_LIMITATIONS.md #3). Provide as list of (T_K, fraction)
    # tuples, e.g. [(1600, 0.7), (1000, 0.2), (400, 0.1)] for a typical
    # hot-dwell / ramp / cold-soak combustor cycle.
    T_schedule: list | None = None

    run_scale1: bool = True
    run_scale2: bool = True
    run_scale3: bool = True
    run_scale4: bool = True
    run_sensitivity: bool = True

    # When True, run_pipeline writes `sobol_indices.csv` into `output_dir`.
    # Defaults False so headless / parallel callers don't race-write the
    # same file. The previous behaviour wrote unconditionally.
    write_sobol_csv: bool = False

    dft_outcar:   str = ""
    lammps_input: str = ""
    # tdb_file points at a CALPHAD database. Default is empty so the
    # pipeline does not attempt to open a non-existent file when CALPHAD
    # is not in use; the previous default ``data/tdb/RE_Si_O.tdb`` was a
    # path that does not exist in the repo and crashed any caller who
    # enabled the CALPHAD branch.
    tdb_file:     str = ""
    output_dir:   str = "results/"


@dataclass
class ScaleParameters:
    """Validated parameters passed between scales."""
    C_ijkl: np.ndarray = field(default_factory=lambda: np.eye(6)*200e9)
    E0: float = 0.0
    V0: float = 0.0
    gamma_surf: float = 1.0
    E_defect: float = 5.0

    kappa_T: np.ndarray = field(default_factory=lambda: np.array([[1.5,0,0],[0,1.5,0],[0,0,1.5]]))
    alpha_T: float = 4.05e-6
    D_O: float = 1e-18
    C_ij_T: np.ndarray = field(default_factory=lambda: np.eye(6)*150e9)

    # Headline effective properties — these alias the EBC sublayer
    # values because Scale 4's bilayer stress lives in the EBC. Use
    # `kappa_eff_TBC` / `E_eff_TBC` for the topcoat and
    # `kappa_eff_EBC` / `E_eff_EBC` for the EBC explicitly; these
    # back-compat scalars are deprecated and may be removed.
    kappa_eff: float = 1.5
    E_eff: float = 150e9
    x_TGO: float = 0.0
    recession: float = 0.0

    # Per-layer stresses — these live in DIFFERENT layers and must NOT be
    # combined arithmetically. Each drives its own potential failure mode.
    sigma_thermal:    float = 0.0   # biaxial stress in the EBC layer
    sigma_TGO_growth: float = 0.0   # biaxial stress in the SiO2 TGO scale

    # Per-interface energy release rates and limit ratios.
    G_drive_EBC:    float = 0.0
    G_drive_TGO:    float = 0.0
    fail_index_EBC: float = 0.0
    fail_index_TGO: float = 0.0

    # Headline scalars: max stress and max failure index across interfaces.
    sigma_max: float = 0.0
    G_drive:   float = 0.0
    fail_index: float = 0.0
    fail_mode: str   = ""           # "EBC" or "TGO" — which interface dominates

    # Diagnostics for the layer-resolved homogenization.
    E_eff_TBC:     float = 0.0
    kappa_eff_TBC: float = 0.0
    E_eff_EBC:     float = 0.0
    kappa_eff_EBC: float = 0.0


def run_pipeline(cfg: TEBCConfig) -> ScaleParameters:
    """Execute full multi-scale pipeline."""
    from tebc.constants import MATERIALS
    params = ScaleParameters()

    # ── SCALE 1 ──
    if cfg.run_scale1 and Path(cfg.dft_outcar).exists():
        logger.info("Scale 1: Parsing DFT outputs...")
        from tebc.scale1_atomistic.dft_interface import parse_elastic_tensor, parse_structure_energy
        params.C_ijkl = parse_elastic_tensor(cfg.dft_outcar)
        struct = parse_structure_energy(cfg.dft_outcar.replace("OUTCAR","vasprun.xml"))
        params.E0 = struct["E0_per_atom"]
        params.V0 = struct["V0_angstrom3"] * 1e-30 / struct["N_atoms"]
        logger.info(f"  C11={params.C_ijkl[0,0]/1e9:.1f} GPa, E0={params.E0:.3f} eV/atom")
    else:
        logger.info("Scale 1: Using database values for %s", cfg.material_EBC)
        mat = MATERIALS[cfg.material_EBC]
        E, nu = mat["E"], mat["nu"]
        lam = E*nu/((1+nu)*(1-2*nu))
        mu  = E/(2*(1+nu))
        C6  = np.zeros((6,6))
        for i in range(3): C6[i,i] = lam + 2*mu
        for i,j in [(0,1),(0,2),(1,2)]: C6[i,j]=C6[j,i]=lam
        for i in range(3,6): C6[i,i] = mu
        params.C_ijkl = C6

    # ── SCALE 2 ──
    if cfg.run_scale2:
        logger.info("Scale 2: MD/QHA thermal properties...")
        mat = MATERIALS[cfg.material_EBC]
        # Mean-section T for elastic softening; hot-face T for the
        # transport processes (oxygen diffusion, TGO oxidation).
        T_mean = (cfg.T_hot + cfg.T_cold) / 2.0
        T_hot  = cfg.T_hot
        if cfg.material_TBC == "7YSZ":
            from tebc.constants import k_B
            ysz = MATERIALS["7YSZ"]
            # D_O drives oxygen flux into the TGO interface — that flux is
            # set by the *hot-face* T, not the section average. Using the
            # average T here previously underpredicted D_O by ~10⁴× at
            # representative T_hot/T_cold.
            params.D_O = (ysz["D0_O"]
                          * np.exp(-ysz["Ea_DO"]/(k_B * T_hot)))
        params.alpha_T  = mat["alpha"]
        params.kappa_T  = np.eye(3) * mat["kappa"]
        # NOTE: this 15 %-over-1300 K linear softening is a placeholder
        # (no thermodynamic basis); flagged in KNOWN_LIMITATIONS.
        params.C_ij_T   = params.C_ijkl * (1 - 0.15*(T_mean-300)/1300)
        logger.info(f"  α={params.alpha_T*1e6:.2f}×10⁻⁶ K⁻¹, κ={params.kappa_T[0,0]:.2f} W/mK")

    # ── SCALE 3 ──
    if cfg.run_scale3:
        logger.info("Scale 3: TGO kinetics + phase-field...")
        from tebc.scale3_mesoscale.tgo_kinetics import (
            parabolic_rate_constant,
            robinson_smialek_recession,
            solve_paralinear,
        )
        mat_bond = MATERIALS[cfg.material_bond]
        T_service = cfg.T_hot

        # Reference-shifted Arrhenius — the database stores k_p as a
        # measured rate at T_ref_kp_wet, NOT an infinite-T prefactor.
        def kp_at(T):
            return parabolic_rate_constant(
                T, mat_bond["k_p_wet"], mat_bond["Ea_kp_wet"],
                T_ref_K=mat_bond["T_ref_kp_wet"],
            )

        def kl_at(T):
            return robinson_smialek_recession(
                T, cfg.P_H2O, cfg.P_tot, cfg.v_gas)

        t_total = cfg.n_cycles * 3600.0
        if cfg.T_schedule:
            from tebc.scale3_mesoscale.tgo_kinetics import (
                integrate_tgo_temperature_schedule,
            )
            sol = integrate_tgo_temperature_schedule(
                t_total, cfg.T_schedule, kp_at, kl_at)
            logger.info(f"  Using duty-cycle schedule: {cfg.T_schedule}")
        else:
            sol = solve_paralinear((0, t_total), kp_at(T_service), kl_at(T_service))
        params.x_TGO   = float(sol["x_TGO"][-1])
        params.recession = float(sol["recession"][-1])

        # Layer-resolved homogenization. The previous code applied a 12 %
        # APS-YSZ porosity to the dense EBC silicate — physically wrong.
        from tebc.coupling.homogenization import (
            maxwell_eucken_kappa, phani_niyogi_modulus,
        )
        # Pore-gas κ rises with T; use a simple linear interpolation
        # between still-air at 300 K (~0.026 W/m·K) and ~0.1 at 1600 K.
        kappa_pore = 0.026 + (0.1 - 0.026) * max(min((T_service-300)/1300, 1.0), 0.0)

        mat_TBC = MATERIALS[cfg.material_TBC]
        mat_EBC = MATERIALS[cfg.material_EBC]
        params.kappa_eff_TBC = maxwell_eucken_kappa(
            mat_TBC["kappa"], kappa_pore, cfg.phi_TBC)
        params.E_eff_TBC     = phani_niyogi_modulus(
            mat_TBC.get("E_APS", mat_TBC["E"]), cfg.phi_TBC)
        params.kappa_eff_EBC = maxwell_eucken_kappa(
            mat_EBC["kappa"], kappa_pore, cfg.phi_EBC)
        params.E_eff_EBC     = phani_niyogi_modulus(
            mat_EBC["E"], cfg.phi_EBC)

        # Back-compat scalar: report the EBC values as the headline E_eff
        # since Scale 4 stress is computed in the EBC layer.
        params.kappa_eff = params.kappa_eff_EBC
        params.E_eff     = params.E_eff_EBC
        logger.info(f"  TGO={params.x_TGO*1e6:.2f} μm, recession={params.recession*1e6:.1f} μm, "
                    f"E_eff_EBC={params.E_eff_EBC/1e9:.1f} GPa, "
                    f"E_eff_TBC={params.E_eff_TBC/1e9:.1f} GPa")

    # ── SCALE 4 ──
    if cfg.run_scale4:
        logger.info("Scale 4: Continuum thermoelastic analysis...")
        from tebc.scale3_mesoscale.tgo_kinetics import tgo_growth_stress
        from tebc.scale4_continuum.thermoelastic import (
            bilayer_mismatch_stress,
            energy_release_rate_steady_state,
        )
        dT = cfg.T_cold - cfg.T_dep
        mat_EBC = MATERIALS[cfg.material_EBC]
        mat_sub = MATERIALS[cfg.material_sub]
        mat_TGO = MATERIALS["SiO2_TGO"]

        # Use the *in-plane* CTE component when available — for monoclinic
        # β-RE silicates, the ab-plane CTE differs from the scalar mean by
        # up to ~3×, which directly scales the bilayer mismatch stress.
        alpha_EBC_inplane = (
            float(mat_EBC["alpha_aniso"][0])
            if "alpha_aniso" in mat_EBC else mat_EBC["alpha"]
        )
        # ---- EBC layer: thermal-mismatch stress and channeling-crack ERR ----
        sigma_thermal = bilayer_mismatch_stress(
            params.E_eff, mat_EBC["nu"],
            alpha_EBC_inplane, mat_sub["alpha"], dT)
        G_EBC = energy_release_rate_steady_state(
            sigma_thermal, cfg.h_EBC, params.E_eff, mat_EBC["nu"])
        Gamma_EBC = mat_EBC["Gamma_interface"]

        # ---- TGO scale: PBR + thermal growth stress, separate ERR ----
        # The TGO is a thin oxide *between* the bond coat and the EBC. Its
        # stress drives delamination at the TGO interface, which is a
        # different failure mode than the EBC channeling crack. We MUST
        # NOT add σ_thermal_EBC and σ_TGO into a single number — they live
        # in different layers separated by the bond coat. Instead each
        # interface gets its own G/Γ ratio and we report the worse one.
        sigma_TGO_elastic = tgo_growth_stress(
            params.x_TGO, mat_TGO["E"], mat_TGO["nu"],
            mat_TGO["alpha"], mat_sub["alpha"], dT,
            PBR=mat_TGO.get("PBR", 2.15),
        )
        # Apply the bond-coat plastic-relaxation knock-down. With
        # tgo_relaxation_factor = 0.07 (default), measured TGO stresses
        # of 1–3 GPa fall out of the ~30 GPa elastic estimate. Set to 1.0
        # to recover the un-relaxed value.
        sigma_TGO = sigma_TGO_elastic * cfg.tgo_relaxation_factor
        G_TGO = energy_release_rate_steady_state(
            sigma_TGO, max(params.x_TGO, 1e-9),
            mat_TGO["E"], mat_TGO["nu"])
        Gamma_TGO = mat_TGO.get("Gamma_interface", 10.0)

        FI_EBC = G_EBC / Gamma_EBC
        FI_TGO = G_TGO / Gamma_TGO
        if FI_EBC >= FI_TGO:
            fail_mode, FI_max = "EBC", FI_EBC
            sigma_dominant, G_dominant = sigma_thermal, G_EBC
        else:
            fail_mode, FI_max = "TGO", FI_TGO
            sigma_dominant, G_dominant = sigma_TGO, G_TGO

        params.sigma_thermal    = sigma_thermal
        params.sigma_TGO_growth = sigma_TGO
        params.G_drive_EBC      = G_EBC
        params.G_drive_TGO      = G_TGO
        params.fail_index_EBC   = FI_EBC
        params.fail_index_TGO   = FI_TGO

        # Headline scalars take the *worse* interface — no spurious sums.
        params.sigma_max  = abs(sigma_dominant)
        params.G_drive    = G_dominant
        params.fail_index = FI_max
        params.fail_mode  = fail_mode
        logger.info(
            f"  σ_th(EBC)={sigma_thermal/1e6:.1f} MPa, "
            f"σ_TGO={sigma_TGO/1e6:.1f} MPa | "
            f"FI_EBC={FI_EBC:.3f}, FI_TGO={FI_TGO:.3f} → "
            f"dominant={fail_mode}",
        )

    # ── SENSITIVITY ──
    if cfg.run_sensitivity:
        logger.info("Sensitivity: Sobol analysis (N=512) on the analytical "
                    "surrogate — see tebc_failure_model docstring; this is "
                    "NOT a sensitivity of run_pipeline itself.")
        from tebc.sensitivity.sobol_morris import run_sobol, tebc_failure_model
        df = run_sobol(tebc_failure_model, N=512)
        logger.info("\n" + df.to_string(index=False))
        if cfg.write_sobol_csv:
            results_dir = Path(cfg.output_dir)
            results_dir.mkdir(exist_ok=True)
            df.to_csv(results_dir / "sobol_indices.csv", index=False)

    return params


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = TEBCConfig(
        run_scale1=False,
        run_scale2=True,
        run_scale3=True,
        run_scale4=True,
        run_sensitivity=True,
    )
    result = run_pipeline(cfg)
    print("\n=== TEBC Simulation Complete ===")
    print(f"Failure index:  {result.fail_index:.4f}  ({'FAIL' if result.fail_index > 1 else 'PASS'})")
    print(f"TGO thickness:  {result.x_TGO*1e6:.2f} μm")
    print(f"EBC recession:  {result.recession*1e6:.2f} μm")
    print(f"Driving force:  {result.G_drive:.1f} J/m²")
