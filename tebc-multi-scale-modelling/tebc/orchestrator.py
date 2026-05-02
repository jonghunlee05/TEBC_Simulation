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
    P_tot:    float = 1.0e5    # Pa  (~ 1 atm)
    v_gas:    float = 10.0     # m/s
    n_cycles: int   = 600

    run_scale1: bool = True
    run_scale2: bool = True
    run_scale3: bool = True
    run_scale4: bool = True
    run_sensitivity: bool = True

    dft_outcar:   str = ""
    lammps_input: str = ""
    tdb_file:     str = "data/tdb/RE_Si_O.tdb"
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

    kappa_eff: float = 1.5
    E_eff: float = 150e9
    x_TGO: float = 0.0
    recession: float = 0.0

    sigma_max: float = 0.0
    G_drive:   float = 0.0
    fail_index: float = 0.0


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
        T_service = (cfg.T_hot + cfg.T_cold) / 2.0
        if cfg.material_TBC == "7YSZ":
            from tebc.constants import k_B
            ysz = MATERIALS["7YSZ"]
            params.D_O = (ysz["D0_O"]
                          * np.exp(-ysz["Ea_DO"]/(k_B * T_service)))
        params.alpha_T  = mat["alpha"]
        params.kappa_T  = np.eye(3) * mat["kappa"]
        params.C_ij_T   = params.C_ijkl * (1 - 0.15*(T_service-300)/1300)
        logger.info(f"  α={params.alpha_T*1e6:.2f}×10⁻⁶ K⁻¹, κ={params.kappa_T[0,0]:.2f} W/mK")

    # ── SCALE 3 ──
    if cfg.run_scale3:
        logger.info("Scale 3: TGO kinetics + phase-field...")
        from tebc.scale3_mesoscale.tgo_kinetics import (
            parabolic_rate_constant,
            robinson_smialek_recession,
            solve_paralinear,
        )
        mat = MATERIALS[cfg.material_bond]
        T_service = cfg.T_hot
        k_p = parabolic_rate_constant(T_service, mat["k_p_wet"], mat["Ea_kp_wet"])
        k_l = robinson_smialek_recession(T_service, cfg.P_H2O, cfg.P_tot, cfg.v_gas)
        t_total = cfg.n_cycles * 3600.0
        sol = solve_paralinear((0, t_total), k_p, k_l)
        params.x_TGO   = float(sol["x_TGO"][-1])
        params.recession = float(sol["recession"][-1])

        from tebc.coupling.homogenization import maxwell_eucken_kappa, phani_niyogi_modulus
        phi_APS = 0.12
        params.kappa_eff = maxwell_eucken_kappa(
            MATERIALS[cfg.material_EBC]["kappa"], 0.025, phi_APS)
        params.E_eff = phani_niyogi_modulus(
            MATERIALS[cfg.material_EBC]["E"], phi_APS)
        logger.info(f"  TGO={params.x_TGO*1e6:.2f} μm, recession={params.recession*1e6:.1f} μm")

    # ── SCALE 4 ──
    if cfg.run_scale4:
        logger.info("Scale 4: Continuum thermoelastic analysis...")
        from tebc.scale4_continuum.thermoelastic import (
            bilayer_mismatch_stress,
            energy_release_rate_steady_state,
        )
        dT = cfg.T_cold - cfg.T_dep
        mat_EBC = MATERIALS[cfg.material_EBC]
        mat_sub = MATERIALS[cfg.material_sub]
        sigma_EBC = bilayer_mismatch_stress(
            params.E_eff, mat_EBC["nu"],
            mat_EBC["alpha"], mat_sub["alpha"], dT)
        params.sigma_max = abs(sigma_EBC)
        params.G_drive   = energy_release_rate_steady_state(
            sigma_EBC, cfg.h_EBC, params.E_eff, mat_EBC["nu"])
        Gamma_int = mat_EBC["Gamma_interface"]
        params.fail_index = params.G_drive / Gamma_int
        logger.info(f"  σ={sigma_EBC/1e6:.1f} MPa, G={params.G_drive:.1f} J/m², FI={params.fail_index:.3f}")

    # ── SENSITIVITY ──
    if cfg.run_sensitivity:
        logger.info("Sensitivity: Sobol analysis (N=512)...")
        from tebc.sensitivity.sobol_morris import run_sobol, tebc_failure_model
        df = run_sobol(tebc_failure_model, N=512)
        logger.info("\n" + df.to_string(index=False))
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
