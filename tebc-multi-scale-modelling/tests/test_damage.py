"""Unit tests for Scale 4 continuum damage mechanics."""

import numpy as np
import pytest

from tebc.scale4_continuum.damage_mechanics import (
    TVHCohesiveZone,
    lemaitre_damage_rate,
    mazars_damage,
    mazars_equivalent_strain,
    triaxiality_factor,
)


def test_triaxiality_uniaxial():
    """For pure uniaxial tension, σ_H/σ_eq = 1/3 → R_v = ⅔(1+ν) + (1-2ν)/3."""
    nu = 0.3
    Rv = triaxiality_factor(sigma_eq=1.0, sigma_H=1 / 3, nu=nu)
    expected = (2.0 / 3.0) * (1 + nu) + (1 - 2 * nu) / 3.0
    assert Rv == pytest.approx(expected, rel=1e-12)


def test_lemaitre_damage_zero_when_undamaged_and_unloaded():
    """No stress → no damage rate, regardless of D."""
    rate = lemaitre_damage_rate(sigma_eq=0.0, sigma_H=0.0, nu=0.3,
                                E=200e9, D=0.5)
    assert rate == 0.0


def test_mazars_zero_below_threshold():
    assert mazars_damage(eps_tilde=1e-5, eps0=1e-4) == 0.0


def test_mazars_saturates_below_one():
    """As ε̃ → ∞, D should approach (but not exceed) 1."""
    D_large = mazars_damage(eps_tilde=1.0)
    assert 0.99 <= D_large <= 0.999


def test_mazars_equivalent_strain_ignores_compression():
    eps = np.array([1e-3, -2e-3, -1e-3])
    et = mazars_equivalent_strain(eps)
    assert et == pytest.approx(1e-3)  # only the positive principal counts


def test_czm_traction_at_origin_is_zero():
    czm = TVHCohesiveZone()
    Tn, Tt = czm.tractions(0.0, 0.0)
    assert (Tn, Tt) == (0.0, 0.0)


def test_czm_traction_at_lambda_one_returns_to_zero():
    """At full separation (λ ≥ 1) traction should be zero."""
    czm = TVHCohesiveZone(sigma_hat=100e6, delta_n_c=1e-6, delta_t_c=3e-6)
    Tn, _ = czm.tractions(delta_n=czm.delta_n_c * 1.5, delta_t=0.0)
    assert Tn == 0.0


def test_czm_Gc_positive():
    czm = TVHCohesiveZone()
    assert czm.G_c > 0
