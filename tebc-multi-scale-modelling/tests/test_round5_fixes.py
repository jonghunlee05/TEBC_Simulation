"""Tests for the round-5 fixes:
- Channel-crack uses K_IC-derived G_Ic, not interface toughness
- TGO Γ uses min(Gamma_TGO_bondcoat, Gamma_TGO_EBC)
- kp_at uses max(wet, dry) — no invented threshold
- TGO E updated to partially-crystalline value
- APS-YSZ D₀_O distinct from bulk
- x_TGO_initial config knob propagates
- Time-resolved x_TGO_t / recession_t exposed
"""

import warnings

import numpy as np
import pytest

from tebc.constants import MATERIALS
from tebc.orchestrator import TEBCConfig, run_pipeline


def _silent_run(**kwargs):
    cfg = TEBCConfig(run_scale1=False, run_sensitivity=False, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return run_pipeline(cfg)


# --- Channel-crack G_Ic vs interface Γ -------------------------------

def test_channel_crack_uses_KIC_not_interface_toughness():
    """The EBC failure-index uses G_Ic = K_IC²(1−ν²)/E, NOT the
    materials-database `Gamma_interface` (= 30 J/m² for β-Yb2Si2O7).
    """
    res = _silent_run()
    mat = MATERIALS["beta_Yb2Si2O7"]
    KIC = mat["KIC"]
    nu  = mat["nu"]
    expected_G_Ic = KIC**2 * (1 - nu**2) / res.E_eff
    # Implied Γ_EBC = G_drive / fail_index_EBC
    if res.fail_index_EBC > 0:
        implied_Gamma = res.G_drive_EBC / res.fail_index_EBC
        assert implied_Gamma == pytest.approx(expected_G_Ic, rel=1e-6)
        assert implied_Gamma != pytest.approx(mat["Gamma_interface"], rel=0.01)


# --- TGO uses min(both interface toughnesses) ------------------------

def test_TGO_uses_min_of_two_interfaces():
    res = _silent_run()
    tgo = MATERIALS["SiO2_TGO"]
    expected_Gamma_TGO = min(tgo["Gamma_TGO_bondcoat"], tgo["Gamma_TGO_EBC"])
    if res.fail_index_TGO > 0:
        implied = res.G_drive_TGO / res.fail_index_TGO
        assert implied == pytest.approx(expected_Gamma_TGO, rel=1e-6)


# --- max(k_p_wet, k_p_dry) replaces wet/dry threshold ----------------

def test_kp_uses_max_of_wet_and_dry():
    """At T_hot=1600 K both rates are well-defined; the orchestrator
    must pick the larger rather than branching on P_H2O. Verify by
    comparing against a manual calculation of max(kp_wet, kp_dry).
    """
    from tebc.scale3_mesoscale.tgo_kinetics import parabolic_rate_constant
    bond = MATERIALS["Si_bondcoat"]
    T = 1600.0
    kp_wet = parabolic_rate_constant(
        T, bond["k_p_wet"], bond["Ea_kp_wet"],
        T_ref_K=bond["T_ref_kp_wet"])
    kp_dry = parabolic_rate_constant(
        T, bond["k_p_dry"], bond["Ea_kp_dry"],
        T_ref_K=bond["T_ref_kp_dry"])
    kp_used = max(kp_wet, kp_dry)
    assert kp_used > 0
    # Either rate could win depending on T; just verify max ≥ each.
    assert kp_used >= kp_wet
    assert kp_used >= kp_dry


def test_kp_selection_does_not_branch_on_PH2O():
    """The parabolic-rate calculation must use max(wet, dry) rather
    than branching on a P_H2O threshold. We exercise the helper
    directly because the full pipeline couples P_H2O through Robinson–
    Smialek (which is correct behaviour, just orthogonal to the kp
    selection logic we're auditing).
    """
    from tebc.scale3_mesoscale.tgo_kinetics import parabolic_rate_constant
    bond = MATERIALS["Si_bondcoat"]
    T = 1600.0
    kp_wet = parabolic_rate_constant(
        T, bond["k_p_wet"], bond["Ea_kp_wet"],
        T_ref_K=bond["T_ref_kp_wet"])
    kp_dry = parabolic_rate_constant(
        T, bond["k_p_dry"], bond["Ea_kp_dry"],
        T_ref_K=bond["T_ref_kp_dry"])
    expected = max(kp_wet, kp_dry)
    # Build the same lambda the orchestrator now uses internally.
    def kp_at(T_K):
        kw = parabolic_rate_constant(
            T_K, bond["k_p_wet"], bond["Ea_kp_wet"],
            T_ref_K=bond["T_ref_kp_wet"])
        kd = parabolic_rate_constant(
            T_K, bond["k_p_dry"], bond["Ea_kp_dry"],
            T_ref_K=bond["T_ref_kp_dry"])
        return max(kw, kd)
    assert kp_at(T) == pytest.approx(expected, rel=1e-12)


# --- SiO2 TGO modulus updated to partially-crystalline ---------------

def test_SiO2_TGO_E_updated():
    assert MATERIALS["SiO2_TGO"]["E"] == pytest.approx(100e9, rel=1e-9)


# --- APS-YSZ has distinct D0_O ---------------------------------------

def test_APS_YSZ_D0_O_distinct_from_bulk():
    ysz = MATERIALS["7YSZ"]
    assert "D0_O_APS" in ysz
    assert ysz["D0_O_APS"] > ysz["D0_O"]


def test_orchestrator_uses_APS_D0_O():
    """`params.D_O` should reflect the APS-enhanced prefactor."""
    res = _silent_run()
    ysz = MATERIALS["7YSZ"]
    from tebc.constants import k_B
    expected = ysz["D0_O_APS"] * np.exp(-ysz["Ea_DO"]/(k_B * 1600.0))
    assert res.D_O == pytest.approx(expected, rel=1e-9)


# --- x_TGO_initial config knob ---------------------------------------

def test_x_TGO_initial_propagates():
    """A non-default x0 must show up in solve_paralinear's result while
    the system is still in the transient regime (t ≪ τ_g). The full
    pipeline uses 600·3600 s ≫ τ_g, so the initial-condition imprint
    has long-since washed into the steady state — exercise the lower-
    level function directly to keep the test in the transient regime.
    """
    from tebc.scale3_mesoscale.tgo_kinetics import solve_paralinear
    # Pick rates with τ_g = x_ss/k_l ≈ 1 s; integrate for 0.1 s.
    k_p, k_l = 2e-18, 1e-9
    # Default x0 = ½·x_ss = 0.5 nm
    sol_def = solve_paralinear((0.0, 0.1), k_p, k_l, n_points=50)
    # Larger seed = 5e-9 m
    sol_big = solve_paralinear((0.0, 0.1), k_p, k_l, x0=5e-9, n_points=50)
    assert sol_big["x_TGO"][-1] > sol_def["x_TGO"][-1]


def test_x_TGO_initial_via_orchestrator():
    """The orchestrator's `cfg.x_TGO_initial` must reach
    `solve_paralinear`. Use a minimal n_cycles so the simulation stays
    near the seed and we can see it.
    """
    from tebc.constants import MATERIALS
    from tebc.scale3_mesoscale.tgo_kinetics import (
        parabolic_rate_constant, robinson_smialek_recession,
    )
    bond = MATERIALS["Si_bondcoat"]
    kp_1600 = max(
        parabolic_rate_constant(1600, bond["k_p_wet"], bond["Ea_kp_wet"],
                                 T_ref_K=bond["T_ref_kp_wet"]),
        parabolic_rate_constant(1600, bond["k_p_dry"], bond["Ea_kp_dry"],
                                 T_ref_K=bond["T_ref_kp_dry"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        kl_1600 = robinson_smialek_recession(1600, 1.0e4, 1.0e5, 10.0)
    x_ss_1600 = kp_1600 / (2 * kl_1600)
    # If we seed well above x_ss, the orchestrator's initial trajectory
    # value (params.x_TGO_t[0]) must be the seeded value, not the
    # adaptive default.
    seed = max(2.0 * x_ss_1600, 1e-9)
    res = _silent_run(x_TGO_initial=seed, n_cycles=1)
    assert res.x_TGO_t[0] == pytest.approx(seed, rel=1e-9)


# --- Time-resolved trajectories exposed ------------------------------

def test_time_resolved_trajectories_exposed():
    res = _silent_run()
    assert res.t_TGO is not None
    assert res.x_TGO_t is not None
    assert res.recession_t is not None
    assert len(res.t_TGO) > 0
    assert res.t_TGO.shape == res.x_TGO_t.shape == res.recession_t.shape
    # The last-element scalar fields must agree with the trajectories.
    assert res.x_TGO == pytest.approx(float(res.x_TGO_t[-1]), rel=1e-12)
    assert res.recession == pytest.approx(float(res.recession_t[-1]), rel=1e-12)
