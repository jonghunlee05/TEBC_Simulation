"""End-to-end checks that run_pipeline now feeds the right numbers into
the failure index: anisotropic CTE, separated EBC/TBC porosities, and a
non-zero TGO growth-stress contribution.
"""

import pytest

from tebc.orchestrator import TEBCConfig, run_pipeline


@pytest.fixture(scope="module")
def base_result():
    cfg = TEBCConfig(
        run_scale1=False,
        run_scale2=True,
        run_scale3=True,
        run_scale4=True,
        run_sensitivity=False,
    )
    return run_pipeline(cfg)


def test_layer_resolved_porosity_yields_distinct_E_eff(base_result):
    """The dense EBC and porous TBC must produce different effective
    moduli now that they're homogenized with their own porosities."""
    assert base_result.E_eff_EBC > base_result.E_eff_TBC
    # Sanity: a 3 % porosity should keep E within ~10 % of bulk.
    assert base_result.E_eff_EBC > 0.85 * 185e9   # bulk β-Yb2Si2O7
    # And a 12 % porosity in APS YSZ should push E_eff well below bulk.
    assert base_result.E_eff_TBC < 0.85 * 50e9


def test_TGO_growth_stress_is_nonzero_and_recorded(base_result):
    """The growth contribution was previously computed in a function but
    never used; it now appears in the dataclass and contributes to σ_max."""
    assert base_result.sigma_TGO_growth != 0.0
    assert base_result.sigma_thermal != 0.0
    assert base_result.sigma_max >= abs(base_result.sigma_thermal)


def test_in_plane_CTE_used_for_mismatch(base_result):
    """The bilayer stress should be computed from `alpha_aniso[0]` (the
    in-plane component), not the scalar `alpha`. For β-Yb2Si2O7 vs SiC
    the two values give different stresses, so the test is just that the
    pipeline matches the in-plane closed form to within float tolerance.
    """
    from tebc.constants import MATERIALS
    from tebc.scale4_continuum.thermoelastic import bilayer_mismatch_stress
    cfg = TEBCConfig()
    mat_EBC = MATERIALS[cfg.material_EBC]
    mat_sub = MATERIALS[cfg.material_sub]
    sigma_inplane_expected = bilayer_mismatch_stress(
        base_result.E_eff,
        mat_EBC["nu"],
        float(mat_EBC["alpha_aniso"][0]),
        mat_sub["alpha"],
        cfg.T_cold - cfg.T_dep,
    )
    assert base_result.sigma_thermal == pytest.approx(
        sigma_inplane_expected, rel=1e-9)

    # And the in-plane value must NOT match what the scalar α would give
    # (regression guard against accidentally reverting to scalar α).
    sigma_scalar = bilayer_mismatch_stress(
        base_result.E_eff, mat_EBC["nu"],
        mat_EBC["alpha"], mat_sub["alpha"],
        cfg.T_cold - cfg.T_dep,
    )
    assert abs(base_result.sigma_thermal - sigma_scalar) > 1e6


def test_failure_index_finite_and_positive(base_result):
    """Pipeline should produce a finite, non-negative failure index."""
    import math
    assert math.isfinite(base_result.fail_index)
    assert base_result.fail_index >= 0.0
