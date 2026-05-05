"""Tests for the round-4 fixes:
- Sobol surrogate aligned with pipeline (PBR^(1/3)-1, relaxation factor)
- Wagner recession (k_l · t / PBR, with transient)
- Hsueh general bilayer formula (Stoney limit + thick-film vanishing)
- TEBCConfig input validation
- HCACF unbiased estimator
- Schedule cycle-period warning
"""

import numpy as np
import pytest

from tebc.scale2_md.green_kubo import compute_hcacf
from tebc.scale3_mesoscale.tgo_kinetics import (
    integrate_tgo_temperature_schedule, solve_paralinear,
)
from tebc.scale4_continuum.thermoelastic import (
    bilayer_mismatch_stress, bilayer_mismatch_stress_hsueh,
)
from tebc.sensitivity.sobol_morris import (
    DEFAULT_TEBC_PROBLEM, tebc_failure_model,
)


# ---- Sobol surrogate alignment ---------------------------------------

def test_surrogate_uses_cube_root_PBR_strain_and_relaxation():
    X = np.array([[1e-6, 1e-13, 30.0, 1.5, 1e-10, 200e9, 0.10]])
    fi_relaxed = tebc_failure_model(X, relaxation_factor=0.07)
    fi_elastic = tebc_failure_model(X, relaxation_factor=1.0)
    assert fi_elastic > fi_relaxed
    fi_pbr_1 = tebc_failure_model(X, PBR=1.0)
    fi_pbr_2 = tebc_failure_model(X, PBR=2.15)
    assert fi_pbr_2 > fi_pbr_1


# ---- Wagner / paralinear recession -----------------------------------

def test_wagner_recession_at_steady_state():
    """At long times in a paralinear regime, dSi/dt → k_l/PBR.

    Pick parameters so growth timescale τ_g = k_p/(2·k_l²) ≪ t_total,
    i.e. x reaches steady state well before the simulation ends.
    """
    k_p, k_l = 2e-18, 1e-9        # x_ss = 1e-9 m, τ_g ≈ 1 s
    sol = solve_paralinear((0.0, 1.0e3), k_p, k_l, n_points=500)
    # At t = 1e3 s we should be deep in steady state.
    assert sol["x_TGO"][-1] == pytest.approx(sol["x_ss"], rel=0.01)
    PBR_sub = 2.15
    expected_late = k_l * sol["t"][-1] / PBR_sub
    assert sol["recession"][-1] == pytest.approx(expected_late, rel=0.05)


def test_recession_smaller_than_old_formula():
    """Wagner recession must be PBR× smaller than the old k_l·t bug."""
    k_p, k_l = 2e-18, 1e-9
    sol = solve_paralinear((0.0, 1.0e3), k_p, k_l, n_points=500)
    old_formula = k_l * sol["t"][-1]
    assert sol["recession"][-1] == pytest.approx(old_formula / 2.15, rel=0.05)


# ---- Hsueh general bilayer ------------------------------------------

def test_hsueh_reduces_to_stoney_in_thin_film_limit():
    args = dict(E_f=200e9, nu_f=0.25, alpha_f=4e-6, alpha_s=10e-6, dT=-1000.0)
    sigma_stoney = bilayer_mismatch_stress(**args)
    sigma_hsueh  = bilayer_mismatch_stress_hsueh(
        h_f=1e-6, h_s=1e-2, E_s=300e9, nu_s=0.20, **args)
    assert sigma_hsueh == pytest.approx(sigma_stoney, rel=1e-3)


def test_hsueh_vanishes_for_thick_film():
    args = dict(E_f=200e9, nu_f=0.25, alpha_f=4e-6, alpha_s=10e-6, dT=-1000.0)
    sigma_hsueh = bilayer_mismatch_stress_hsueh(
        h_f=1e-2, h_s=1e-6, E_s=300e9, nu_s=0.20, **args)
    sigma_stoney = bilayer_mismatch_stress(**args)
    assert abs(sigma_hsueh) < 0.01 * abs(sigma_stoney)


# ---- TEBCConfig input validation -------------------------------------

def test_config_rejects_unphysical_temperatures():
    from tebc.orchestrator import TEBCConfig
    with pytest.raises(ValueError, match="Inconsistent or unphysical"):
        TEBCConfig(T_hot=5000.0)
    with pytest.raises(ValueError, match="Inconsistent or unphysical"):
        TEBCConfig(T_hot=300.0, T_cold=400.0)


def test_config_rejects_negative_n_cycles():
    from tebc.orchestrator import TEBCConfig
    with pytest.raises(ValueError, match="n_cycles"):
        TEBCConfig(n_cycles=-5)


def test_config_rejects_bad_porosity():
    from tebc.orchestrator import TEBCConfig
    with pytest.raises(ValueError, match="phi_TBC"):
        TEBCConfig(phi_TBC=1.5)


def test_config_rejects_bad_T_schedule():
    from tebc.orchestrator import TEBCConfig
    with pytest.raises(ValueError, match="T_schedule"):
        TEBCConfig(T_schedule=[(1600.0, 0.5), (1000.0, 0.6)])
    with pytest.raises(ValueError, match="T_schedule"):
        TEBCConfig(T_schedule=[(-100.0, 1.0)])


def test_config_rejects_bad_relaxation_factor():
    from tebc.orchestrator import TEBCConfig
    with pytest.raises(ValueError, match="tgo_relaxation_factor"):
        TEBCConfig(tgo_relaxation_factor=2.0)


# ---- HCACF estimator -------------------------------------------------

def test_hcacf_unbiased_amplifies_long_lags():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(2000, 3))
    _, biased = compute_hcacf(J, dt=1.0, unbiased=False)
    _, unb    = compute_hcacf(J, dt=1.0, unbiased=True)
    assert biased[0] == pytest.approx(unb[0], rel=1e-12)
    long_lag = len(biased) // 2
    assert abs(unb[long_lag]) >= abs(biased[long_lag])


# ---- Schedule cycle-period warning -----------------------------------

def test_schedule_warns_when_cycle_period_long():
    def kp_at(T): return 1e-18
    def kl_at(T): return 1e-12
    with pytest.warns(UserWarning, match="growth timescale"):
        integrate_tgo_temperature_schedule(
            t_total=1e9,
            schedule=[(1600.0, 0.5), (1000.0, 0.5)],
            k_p_at=kp_at, k_l_at=kl_at,
            cycle_period=1e7,
        )


# ---- Sobol problem unchanged ----------------------------------------

def test_sobol_problem_dimensions_unchanged():
    p = DEFAULT_TEBC_PROBLEM
    assert p["num_vars"] == 7
    assert len(p["names"]) == 7
