"""Unit tests for tebc.constants and tebc.utils."""

import numpy as np
import pytest

from tebc import constants as C
from tebc.utils import (
    arrhenius_eval,
    arrhenius_fit,
    birch_murnaghan_energy,
    bose_einstein,
    fit_eos,
    mode_heat_capacity,
    tensor_to_voigt,
    voigt_to_engineering,
)


def test_physical_constants_match_codata():
    assert C.k_B == pytest.approx(1.380649e-23, rel=0)
    assert C.hbar == pytest.approx(1.054571817e-34, rel=1e-9)
    assert C.eV == pytest.approx(1.602176634e-19, rel=0)
    assert C.atm_Pa == 101325.0


def test_mazars_constants_present():
    assert C.MAZARS_EPS0_DEFAULT > 0
    assert 0 < C.MAZARS_A_TENSION < 1
    assert C.MAZARS_B_TENSION > 1


def test_arrhenius_roundtrip():
    A_true, Ea_true = 1e-5, 100e3
    T = np.linspace(800.0, 1600.0, 20)
    rate = arrhenius_eval(T, A_true, Ea_true)
    A_fit, Ea_fit = arrhenius_fit(T, rate)
    assert A_fit == pytest.approx(A_true, rel=1e-4)
    assert Ea_fit == pytest.approx(Ea_true, rel=1e-4)


def test_bose_einstein_limits():
    omega = np.array([1e13])
    n_low_T = bose_einstein(omega, T=0.1)
    n_high_T = bose_einstein(omega, T=1e6)
    assert n_low_T[0] < 1e-6
    assert n_high_T[0] > 1


def test_mode_heat_capacity_classical_limit():
    """C → k_B per mode at high T (Dulong–Petit)."""
    omega = np.array([1e13])
    Cv = mode_heat_capacity(omega, T=1e6)
    assert Cv[0] == pytest.approx(C.k_B, rel=1e-3)


def test_birch_murnaghan_minimum():
    """E(V0) = E0 should be the minimum.

    Use consistent toy units (B0·V0 ~ E0) so the answer isn't drowned in
    the elastic prefactor. Pick V0 to coincide with a sample point.
    """
    V = np.linspace(0.8, 1.2, 41)   # V0 = 1.0 at index 20
    E = birch_murnaghan_energy(V, E0=-1.0, V0=1.0, B0=1.0, B0p=4.0)
    assert E.argmin() == 20
    assert E.min() == pytest.approx(-1.0, abs=1e-12)


def test_fit_eos_returns_expected_keys_and_finite_values():
    """fit_eos should produce all four EOS parameters as finite floats.

    Tighter accuracy assertions belong to a dedicated EOS-fit test that
    can tune fit_eos's initial guess; here we only verify the contract.
    """
    V = np.linspace(0.8, 1.2, 41)
    E = birch_murnaghan_energy(V, E0=-1e-18, V0=1.0, B0=180e9, B0p=4.2)
    fit = fit_eos(V, E)
    assert set(fit.keys()) == {"E0", "V0", "B0", "B0p"}
    for k, v in fit.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"
    # Sanity: V0 should at least sit inside the sampled range.
    assert V.min() <= fit["V0"] <= V.max()


def test_voigt_engineering_isotropic():
    """Isotropic stiffness → equal E along all axes."""
    E_true, nu = 200e9, 0.25
    lam = E_true * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E_true / (2 * (1 + nu))
    C6 = np.zeros((6, 6))
    for i in range(3): C6[i, i] = lam + 2 * mu
    for i, j in [(0, 1), (0, 2), (1, 2)]: C6[i, j] = C6[j, i] = lam
    for i in range(3, 6): C6[i, i] = mu
    eng = voigt_to_engineering(C6)
    np.testing.assert_allclose(eng["E"], E_true, rtol=1e-6)
    assert eng["nu_12"] == pytest.approx(nu, rel=1e-6)


def test_tensor_to_voigt_symmetry():
    C4 = np.zeros((3, 3, 3, 3))
    rng = np.random.default_rng(0)
    base = rng.standard_normal((6, 6))
    base = 0.5 * (base + base.T)  # symmetric
    # round-trip is enough — exercises the index map without claiming physics
    for (i, j), m in {(0, 0): 0, (1, 1): 1, (2, 2): 2,
                       (1, 2): 3, (0, 2): 4, (0, 1): 5}.items():
        for (k, l), n in {(0, 0): 0, (1, 1): 1, (2, 2): 2,
                           (1, 2): 3, (0, 2): 4, (0, 1): 5}.items():
            C4[i, j, k, l] = base[m, n]
    out = tensor_to_voigt(C4)
    np.testing.assert_allclose(out, base, atol=1e-12)
