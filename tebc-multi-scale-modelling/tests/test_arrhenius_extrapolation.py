"""Regression tests for the parabolic-rate Arrhenius extrapolation.

Old behaviour: parabolic_rate_constant(T, k_ref, Ea) treated `k_ref` as
the infinite-temperature prefactor A and evaluated A·exp(-Ea/RT). When
the database actually stores a *measured rate at a finite reference
T_ref*, this under-predicts k(T) by exp(Ea/(R·T_ref)) — typically a
factor of 10²–10⁴ for oxidation kinetics.

New behaviour (with `T_ref_K` given): reference-shifted Arrhenius
    k(T) = k_ref · exp(-Ea/R · (1/T - 1/T_ref))
which by construction satisfies k(T_ref) = k_ref.
"""

import numpy as np
import pytest

from tebc.constants import MATERIALS, R_gas
from tebc.scale3_mesoscale.tgo_kinetics import parabolic_rate_constant


def test_kp_at_reference_T_returns_kref():
    """k(T_ref) must equal k_ref exactly when using the shifted form."""
    bond = MATERIALS["Si_bondcoat"]
    k = parabolic_rate_constant(
        bond["T_ref_kp_wet"], bond["k_p_wet"], bond["Ea_kp_wet"],
        T_ref_K=bond["T_ref_kp_wet"],
    )
    assert k == pytest.approx(bond["k_p_wet"], rel=1e-12)


def test_kp_increases_with_T():
    """k(T) is monotonically increasing for Ea > 0."""
    bond = MATERIALS["Si_bondcoat"]
    Ts = np.array([1200.0, 1400.0, 1600.0, 1800.0])
    ks = np.array([
        parabolic_rate_constant(T, bond["k_p_wet"], bond["Ea_kp_wet"],
                                T_ref_K=bond["T_ref_kp_wet"])
        for T in Ts
    ])
    assert np.all(np.diff(ks) > 0)


def test_kp_old_call_underpredicts_at_high_T():
    """The legacy call (no T_ref) treats k_ref as A and therefore under-
    predicts k(T) by a multiplicative exp(Ea/(R·T_ref)). The size of the
    correction depends on Ea and T_ref; for β-Yb2Si2O7 at 1316 °C with
    Ea ≈ 101 kJ/mol the correction is >1000×."""
    yb = MATERIALS["beta_Yb2Si2O7"]
    T_ref = yb["T_ref_kp"]
    Ea    = yb["Ea_kp"]
    k_ref = yb["k_p_TGO"]
    T = 1700.0

    k_correct = parabolic_rate_constant(T, k_ref, Ea, T_ref_K=T_ref)
    k_legacy  = parabolic_rate_constant(T, k_ref, Ea)               # no T_ref
    ratio = k_correct / k_legacy

    expected_ratio = np.exp(Ea / (R_gas * T_ref))
    assert ratio == pytest.approx(expected_ratio, rel=1e-9)
    assert ratio > 1e3, f"expected ≥10³× correction; got {ratio:.2g}"


def test_si_bondcoat_correction_is_order_100():
    """Si bond coat wet oxidation has a smaller Ea ≈ 68 kJ/mol; the
    legacy/correct ratio is ~170× rather than 10³. Still a serious bug."""
    bond = MATERIALS["Si_bondcoat"]
    T_ref = bond["T_ref_kp_wet"]
    Ea    = bond["Ea_kp_wet"]
    k_ref = bond["k_p_wet"]
    k_correct = parabolic_rate_constant(1600.0, k_ref, Ea, T_ref_K=T_ref)
    k_legacy  = parabolic_rate_constant(1600.0, k_ref, Ea)
    assert 100 < k_correct / k_legacy < 500
