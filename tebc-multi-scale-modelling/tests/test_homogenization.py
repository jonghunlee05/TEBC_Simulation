"""Unit tests for cross-scale homogenization schemes."""

import numpy as np
import pytest

from tebc.coupling.homogenization import (
    hashin_shtrikman_bulk_modulus,
    hill_average,
    maxwell_eucken_kappa,
    mori_tanaka_spheres,
    phani_niyogi_modulus,
    reuss_average,
    voigt_average,
)


def _iso_stiffness(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    for i in range(3): C[i, i] = lam + 2 * mu
    for i, j in [(0, 1), (0, 2), (1, 2)]: C[i, j] = C[j, i] = lam
    for i in range(3, 6): C[i, i] = mu
    return C


def test_voigt_reuss_single_phase_identity():
    """One phase at f=1 should reproduce its own stiffness exactly."""
    C = _iso_stiffness(200e9, 0.25)
    np.testing.assert_allclose(voigt_average([C], [1.0]), C)
    np.testing.assert_allclose(reuss_average([C], [1.0]), C)


def test_voigt_reuss_hill_ordering():
    """Voigt is upper, Reuss is lower, Hill in the middle (trace test)."""
    C1 = _iso_stiffness(200e9, 0.25)
    C2 = _iso_stiffness(50e9, 0.30)
    f = [0.6, 0.4]
    tr_V = np.trace(voigt_average([C1, C2], f))
    tr_R = np.trace(reuss_average([C1, C2], f))
    tr_H = np.trace(hill_average([C1, C2], f))
    assert tr_V >= tr_H >= tr_R


def test_hashin_shtrikman_bounds_bracket_voigt_reuss():
    """HS bounds should sit between Voigt and Reuss bulk moduli."""
    K1, K2 = 100e9, 200e9
    G1, G2 = 40e9, 80e9
    f1, f2 = 0.4, 0.6
    K_lo, K_hi = hashin_shtrikman_bulk_modulus(K1, K2, G1, G2, f1, f2)
    K_voigt = f1 * K1 + f2 * K2
    K_reuss = 1 / (f1 / K1 + f2 / K2)
    assert K_reuss <= K_lo <= K_hi <= K_voigt


def test_mori_tanaka_dilute_limit():
    """At f_i → 0 the effective moduli → matrix moduli."""
    res = mori_tanaka_spheres(K_m=100e9, G_m=40e9,
                              K_i=200e9, G_i=80e9, f_i=1e-6)
    assert res["K_eff"] == pytest.approx(100e9, rel=1e-4)
    assert res["G_eff"] == pytest.approx(40e9, rel=1e-4)


def test_maxwell_eucken_dense_and_porous_limits():
    assert maxwell_eucken_kappa(2.5, 0.0, 0.0) == pytest.approx(2.5, rel=1e-12)
    # κ should drop monotonically with porosity
    phis = [0.0, 0.1, 0.2, 0.3]
    ks = [maxwell_eucken_kappa(2.5, 0.025, p) for p in phis]
    assert all(ks[i] >= ks[i + 1] for i in range(len(phis) - 1))


def test_phani_niyogi_zero_at_critical_porosity():
    E0 = 200e9
    assert phani_niyogi_modulus(E0, phi=0.6, phi_c=0.6) == 0.0
    assert phani_niyogi_modulus(E0, phi=0.0) == pytest.approx(E0)
    # Beyond φ_c clamps to zero, never negative.
    assert phani_niyogi_modulus(E0, phi=0.8, phi_c=0.6) == 0.0
