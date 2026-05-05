"""Tests for cyclic-temperature TGO integration and PBR strain.

KNOWN_LIMITATIONS #3 used to say: TGO is integrated at constant T_hot,
overestimating growth by orders of magnitude. The new
`integrate_tgo_temperature_schedule` weights k_p and k_l by a duty
cycle, which gives a much smaller — and physically reasonable —
effective rate when most of the cycle is spent at intermediate T.
"""

import numpy as np
import pytest

from tebc.constants import MATERIALS
from tebc.scale3_mesoscale.tgo_kinetics import (
    integrate_tgo_temperature_schedule,
    parabolic_rate_constant,
    robinson_smialek_recession,
    solve_paralinear,
    tgo_growth_stress,
)


def _build_rate_callables():
    bond = MATERIALS["Si_bondcoat"]
    def kp_at(T):
        return parabolic_rate_constant(
            T, bond["k_p_wet"], bond["Ea_kp_wet"],
            T_ref_K=bond["T_ref_kp_wet"])
    def kl_at(T):
        # Suppress the out-of-domain warning (v_gas = 10 m/s is far
        # above the R–S anchor of 4.4 cm/s; that's not what we're
        # testing here — we're testing the duty-cycle integration).
        return robinson_smialek_recession(
            T, 1.0e4, 1.0e5, 10.0, warn_out_of_domain=False)
    return kp_at, kl_at


def test_pbr_strain_is_cube_root_form():
    """ε_growth must be PBR^(1/3)−1, not (PBR−1)/3.

    For SiO2 on Si (PBR = 2.15) the two differ by ~30 %; we check the
    sign and magnitude of σ_growth match the cube-root expectation.
    """
    sigma = tgo_growth_stress(
        x_TGO=1e-6, E_TGO=70e9, nu_TGO=0.17,
        alpha_TGO=0.55e-6, alpha_sub=4.8e-6, dT=0.0,    # zero ΔT isolates growth
        PBR=2.15,
    )
    biaxial = 70e9 / (1 - 0.17)
    eps_correct = 2.15 ** (1/3) - 1.0
    expected_sigma = -biaxial * eps_correct
    assert sigma == pytest.approx(expected_sigma, rel=1e-12)


def test_schedule_with_full_dwell_at_T_hot_matches_isothermal():
    """Schedule with 100 % at T_hot must equal a plain isothermal solve."""
    kp_at, kl_at = _build_rate_callables()
    t_total = 600.0 * 3600.0
    sol_sched = integrate_tgo_temperature_schedule(
        t_total, [(1600.0, 1.0)], kp_at, kl_at, n_points=200)
    sol_iso = solve_paralinear(
        (0.0, t_total), kp_at(1600.0), kl_at(1600.0), n_points=200)
    assert sol_sched["x_TGO"][-1] == pytest.approx(sol_iso["x_TGO"][-1], rel=1e-9)


def test_schedule_lowers_effective_kp():
    """Adding cold-T fractions must drop the effective parabolic rate.

    Note that the *final TGO thickness* may not strictly decrease,
    because for our Ea_kp / Ea_kl ratio (68 / 108 kJ/mol) the linear
    rate drops faster than the parabolic at lower T, raising the
    paralinear steady state x_ss = k_p/(2 k_l). The physically
    meaningful invariant is that the effective transient rate is lower.
    """
    kp_at, kl_at = _build_rate_callables()
    t_total = 600.0 * 3600.0
    hot_only = integrate_tgo_temperature_schedule(
        t_total, [(1600.0, 1.0)], kp_at, kl_at, n_points=200)
    realistic = integrate_tgo_temperature_schedule(
        t_total, [(1600.0, 0.7), (1000.0, 0.2), (400.0, 0.1)],
        kp_at, kl_at, n_points=200)
    assert realistic["effective_k_p"] < 0.8 * hot_only["effective_k_p"]
    assert realistic["effective_k_l"] < 0.8 * hot_only["effective_k_l"]


def test_schedule_fractions_must_sum_to_one():
    kp_at, kl_at = _build_rate_callables()
    with pytest.raises(ValueError, match="sum to 1"):
        integrate_tgo_temperature_schedule(
            1000.0, [(1600.0, 0.5), (1000.0, 0.3)], kp_at, kl_at)
