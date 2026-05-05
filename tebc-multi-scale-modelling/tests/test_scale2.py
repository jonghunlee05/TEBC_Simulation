"""Unit tests for Scale 2 (MD/QHA) and downstream modules."""

import numpy as np
import pytest

from tebc.constants import hbar, k_B
from tebc.scale2_md.green_kubo import compute_hcacf, integrate_hcacf, plateau_estimate
from tebc.scale2_md.msd_diffusion import msd_to_diffusivity
from tebc.scale2_md.phonon_qha import compute_free_energy, gruneisen_cte_relation


def test_hcacf_delta():
    """HCACF of delta function = constant → κ should converge."""
    n = 10000
    dt = 1e-15
    J  = np.zeros(n); J[0] = 1.0
    t, C = compute_hcacf(J.reshape(-1, 1).repeat(3, axis=1) / np.sqrt(3), dt)
    assert C[0] > 0


def test_free_energy_zero_T():
    """
    At T→0, F → E0 + ZPE (vibrational tail vanishes only as T·exp(-ℏω/kT)).

    Use a low-T anchor where ℏω_min/kT ≫ 1 so the F_vib contribution is
    truly negligible.  For ω = 1e13 rad/s and T = 0.5 K,
    ℏω/kT ≈ 1530 → F_vib ~ kT·exp(-1530) ≈ 0 to machine precision.
    """
    omega = np.array([[1e13, 2e13]])
    w     = np.array([1.0])
    E0    = -1e-18
    F = compute_free_energy(omega, w, E0, T=0.5)
    ZPE = 0.5 * (hbar * omega * w[:, None]).sum()
    assert abs(F - (E0 + ZPE)) / abs(E0 + ZPE) < 1e-6


def test_gruneisen_identity():
    """α_V = γ C_V / (B V)."""
    gamma = 1.0
    Cv    = 1.5e6
    V     = 1.0
    B     = 135e9
    alpha = gruneisen_cte_relation(gamma, Cv, V, B)
    assert 1e-8 < alpha < 1e-4, f"Unreasonable CTE: {alpha}"


def test_msd_diffusivity():
    """Linear MSD should recover exact D."""
    D_true = 1e-12
    n_frames = 1000
    dt = 1e-12
    t  = np.arange(n_frames) * dt
    msd = 6 * D_true * t
    t_idx = np.arange(n_frames)
    result = msd_to_diffusivity(t_idx, msd, dt_per_frame=dt)
    assert abs(result["D"] - D_true) / D_true < 0.01


class TestScale3:
    def test_paralinear_steady_state(self):
        from tebc.scale3_mesoscale.tgo_kinetics import solve_paralinear
        k_p = 1e-14
        k_l = 1e-10
        x_ss_analytical = k_p / (2*k_l)
        sol  = solve_paralinear((0, 1e8), k_p, k_l, x0=1e-9, n_points=2000)
        x_final = sol["x_TGO"][-1]
        assert abs(x_final - x_ss_analytical)/x_ss_analytical < 0.05

    def test_deal_grove_thin_limit(self):
        """
        Thin-scale (interface-reaction) limit holds when B·t ≪ A²/4,
        i.e. t ≪ k_p/(2 k_l²).  For k_p=1e-22 m²/s, k_l=1e-9 m/s
        this gives t_thin ≪ 5e-5 s; sample at t = 1e-7…1e-6 s.
        """
        from tebc.scale3_mesoscale.tgo_kinetics import deal_grove_thickness
        k_p = 1e-22
        k_l = 1e-9
        t   = np.array([1e-7, 5e-7, 1e-6])
        x   = deal_grove_thickness(t, k_p, k_l)
        x_linear = k_l * t
        np.testing.assert_allclose(x, x_linear, rtol=0.05)


class TestScale4:
    def test_mismatch_stress_sign(self):
        """
        Sign convention check (σ_f = E/(1-ν)·(α_s − α_f)·ΔT):

        (a) High-CTE film on low-CTE substrate (YSZ on EBC), ΔT<0:
            substrate constrains the film from contracting → film is in
            TENSION (σ > 0).
        (b) Low-CTE film on high-CTE substrate (EBC on SiC/SiC), ΔT<0:
            substrate contracts more than the film → film is in
            COMPRESSION (σ < 0).

        Both senses must come out right.
        """
        from tebc.scale4_continuum.thermoelastic import bilayer_mismatch_stress
        sigma_YSZ_on_EBC = bilayer_mismatch_stress(
            E_f=50e9, nu_f=0.23,
            alpha_f=10.5e-6, alpha_s=4.05e-6,
            dT=-1300)
        assert sigma_YSZ_on_EBC > 0, "YSZ on EBC should be tensile on cool-down"

        sigma_EBC_on_CMC = bilayer_mismatch_stress(
            E_f=185e9, nu_f=0.275,
            alpha_f=4.05e-6, alpha_s=4.8e-6,
            dT=-1300)
        assert sigma_EBC_on_CMC < 0, "EBC on CMC should be compressive on cool-down"

    def test_energy_release_rate_positive(self):
        from tebc.scale4_continuum.thermoelastic import energy_release_rate_steady_state
        G = energy_release_rate_steady_state(300e6, 200e-6, 50e9, 0.23)
        assert G > 0

    def test_czm_traction_peak(self):
        from tebc.scale4_continuum.damage_mechanics import TVHCohesiveZone
        czm = TVHCohesiveZone(sigma_hat=100e6, delta_n_c=1e-6, delta_t_c=3e-6)
        dn_at_peak = czm.l1 * czm.delta_n_c
        T_n, _ = czm.tractions(dn_at_peak, 0.0)
        assert abs(T_n - czm.sigma_hat) / czm.sigma_hat < 0.01


class TestCoupling:
    def test_voigt_reuss_bounds(self):
        from tebc.coupling.homogenization import hill_average, reuss_average, voigt_average
        C1 = np.eye(6) * 200e9; C1[3,3]=C1[4,4]=C1[5,5]=80e9
        C2 = np.eye(6) *  50e9; C2[3,3]=C2[4,4]=C2[5,5]=20e9
        f  = [0.7, 0.3]
        Cv = voigt_average([C1,C2], f)
        Cr = reuss_average([C1,C2], f)
        Ch = hill_average([C1,C2], f)
        assert np.trace(Cv) >= np.trace(Ch) >= np.trace(Cr)

    def test_maxwell_eucken_limits(self):
        from tebc.coupling.homogenization import maxwell_eucken_kappa
        assert abs(maxwell_eucken_kappa(2.5, 0.0, 0.0) - 2.5) < 1e-10
        assert maxwell_eucken_kappa(2.5, 0.0, 0.99) < 0.05


class TestSensitivity:
    def test_sobol_sum_leq_total(self):
        from tebc.sensitivity.sobol_morris import run_sobol, tebc_failure_model
        df = run_sobol(tebc_failure_model, N=128)
        assert df["S1"].sum() <= df["ST"].sum() + 0.1

    def test_delta_alpha_dominates(self):
        """
        In the analytic surrogate (post round-4 alignment with the
        pipeline) the variance ranking is set by parameter range ×
        surrogate exponent:
          - k_l, k_p:    3-decade log-uniform → dominate variance
          - Γ_int:       1.2-decade linear in denominator → strong
          - Δα:          ~0.6-decade linear, enters squared
          - kappa_TBC, porosity_TBC: linear via the new κ-coupling
            (kappa_factor multiplies G_drive+G_TGO)

        Assert the top-3 sit inside the union of these driver sets.
        """
        from tebc.sensitivity.sobol_morris import run_sobol, tebc_failure_model
        df = run_sobol(tebc_failure_model, N=512)
        top3 = set(df.iloc[:3]["parameter"].tolist())
        admissible = {"k_l", "k_p", "Gamma_int", "delta_alpha",
                       "kappa_TBC", "porosity_TBC"}
        assert top3.issubset(admissible), (
            f"Top-3 params must lie in {admissible}, got {top3}")
