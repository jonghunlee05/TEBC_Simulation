"""Tests for the small polish fixes:
- TGO viscoplastic relaxation factor
- Sobol CSV write is opt-in
- Robinson–Smialek out-of-domain warning
- MSD framework_mask defaults to ~species_mask
- phani_niyogi_modulus φ_c default lowered to 0.45
- cahill_pohl_kappa_min per-branch Θ_D
- solve_paralinear convergence flag
- fenics_thermoelastic_setup removed
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from tebc.coupling.homogenization import (
    cahill_pohl_kappa_min, phani_niyogi_modulus,
)
from tebc.orchestrator import TEBCConfig, run_pipeline
from tebc.scale2_md.msd_diffusion import compute_msd
from tebc.scale3_mesoscale.tgo_kinetics import (
    robinson_smialek_recession, solve_paralinear,
)


def test_tgo_relaxation_factor_scales_growth_stress():
    """Halving the relaxation factor should halve σ_TGO_growth."""
    cfg_full = TEBCConfig(run_scale1=False, run_sensitivity=False,
                           tgo_relaxation_factor=1.0)
    cfg_half = TEBCConfig(run_scale1=False, run_sensitivity=False,
                           tgo_relaxation_factor=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = run_pipeline(cfg_full)
        r2 = run_pipeline(cfg_half)
    assert r2.sigma_TGO_growth == pytest.approx(0.5 * r1.sigma_TGO_growth, rel=1e-9)


def test_sobol_csv_not_written_by_default(tmp_path):
    """run_pipeline must not race-write sobol_indices.csv unless asked."""
    cfg = TEBCConfig(
        run_scale1=False, run_scale2=True, run_scale3=True,
        run_scale4=True, run_sensitivity=True,
        write_sobol_csv=False,
        output_dir=str(tmp_path),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_pipeline(cfg)
    assert not (tmp_path / "sobol_indices.csv").exists()


def test_sobol_csv_written_when_opted_in(tmp_path):
    cfg = TEBCConfig(
        run_scale1=False, run_scale2=True, run_scale3=True,
        run_scale4=True, run_sensitivity=True,
        write_sobol_csv=True,
        output_dir=str(tmp_path),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_pipeline(cfg)
    assert (tmp_path / "sobol_indices.csv").exists()


def test_robinson_smialek_warns_out_of_domain():
    """v_gas = 10 m/s is ~227× the calibration anchor — must warn."""
    with pytest.warns(UserWarning, match="outside the Opila"):
        robinson_smialek_recession(1600.0, 1e4, 1e5, 10.0)


def test_robinson_smialek_quiet_in_domain():
    """At calibration anchors no warning should be raised."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Anchors: T=1589 K, P_H2O = 0.1 atm, v=4.4 cm/s.
        robinson_smialek_recession(1589.0, 0.1*101325.0, 1e5, 0.044)


def test_msd_framework_default_excludes_species():
    """When framework_mask is None and species_mask is given,
    framework should default to ~species_mask (not all atoms)."""
    rng = np.random.default_rng(0)
    n_frames, n_atoms = 1000, 20
    species = np.zeros(n_atoms, dtype=bool)
    species[:5] = True   # first 5 atoms are the diffusing species

    # Build a trajectory where the diffusing species drifts and the
    # framework is stationary; the new default should remove only the
    # framework (zero) drift, leaving the species' motion intact.
    pos = np.zeros((n_frames, n_atoms, 3))
    drift = np.linspace(0, 1.0, n_frames)[:, None]
    pos[:, species, :] = drift[:, None, :]
    # tiny jitter
    pos += rng.normal(0, 1e-6, size=pos.shape)

    _, msd = compute_msd(pos, species_mask=species, remove_com_drift=True)
    # The species moved by ~1.0 m → MSD at long lag should be ~1.0².
    assert msd[-1] > 0.5


def test_phani_niyogi_default_phic_lowered():
    """At φ = 0.20 (typical APS) the default φ_c = 0.45 retains less
    stiffness than the old 0.6 default would have."""
    E0 = 200e9
    new = phani_niyogi_modulus(E0, phi=0.20)               # default φ_c=0.45
    old = phani_niyogi_modulus(E0, phi=0.20, phi_c=0.6)    # legacy
    assert new < old
    assert 0.30 * E0 < new < 0.40 * E0
    assert 0.42 * E0 < old < 0.48 * E0


def test_cahill_pohl_per_branch_theta_default():
    """When `theta_D=None`, Θ_i is computed from each sound speed; the
    result should differ from the scalar-Θ form for non-uniform speeds."""
    n = 5e28
    v = np.array([6000.0, 3500.0, 3500.0])    # 1 longitudinal, 2 transverse
    k_per_branch = cahill_pohl_kappa_min(2.0, n, v, T=1000.0, theta_D=None)
    k_scalar     = cahill_pohl_kappa_min(2.0, n, v, T=1000.0, theta_D=400.0)
    assert k_per_branch > 0
    assert k_scalar > 0
    assert k_per_branch != pytest.approx(k_scalar, rel=1e-3)


def test_solve_paralinear_reports_success_flag():
    sol = solve_paralinear((0.0, 1e6), k_p=1e-18, k_l=1e-12, n_points=100)
    assert sol["success"] is True


def test_fenics_setup_function_removed():
    """The misleading string-template function must be gone."""
    from tebc.scale4_continuum import thermoelastic
    assert not hasattr(thermoelastic, "fenics_thermoelastic_setup")
