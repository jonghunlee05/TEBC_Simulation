"""End-to-end checks that run_pipeline now feeds the right numbers into
the failure index: anisotropic CTE, separated EBC/TBC porosities, and a
non-zero TGO growth-stress contribution.
"""

import pytest

from tebc.orchestrator import TEBCConfig, run_pipeline


@pytest.fixture(scope="module")
def base_result():
    import warnings
    cfg = TEBCConfig(
        run_scale1=False,
        run_scale2=True,
        run_scale3=True,
        run_scale4=True,
        run_sensitivity=False,
    )
    # Suppress the v_gas-out-of-domain warning that fires on every
    # default-config call; it's about the calibration window, not what
    # this fixture is testing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
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
    never used; it now appears in the dataclass."""
    assert base_result.sigma_TGO_growth != 0.0
    assert base_result.sigma_thermal != 0.0


def test_per_interface_failure_indices_are_separate(base_result):
    """σ_thermal lives in the EBC and σ_TGO in the TGO scale; the two
    must not be combined into a single ERR. Each interface should have
    its own G_drive and FI, and the headline values should pick the
    worse of the two — not sum them.
    """
    import math
    assert math.isfinite(base_result.G_drive_EBC)
    assert math.isfinite(base_result.G_drive_TGO)
    assert math.isfinite(base_result.fail_index_EBC)
    assert math.isfinite(base_result.fail_index_TGO)

    # Both per-interface FIs are non-negative.
    assert base_result.fail_index_EBC >= 0.0
    assert base_result.fail_index_TGO >= 0.0

    # Headline FI equals the max of the two — and is *not* their sum.
    fi_max = max(base_result.fail_index_EBC, base_result.fail_index_TGO)
    fi_sum = base_result.fail_index_EBC + base_result.fail_index_TGO
    assert base_result.fail_index == pytest.approx(fi_max, rel=1e-12)
    if base_result.fail_index_EBC > 0 and base_result.fail_index_TGO > 0:
        assert base_result.fail_index < fi_sum, (
            "Headline FI should not be the arithmetic sum of per-interface "
            "FIs — that was the bug the previous commit introduced."
        )

    # fail_mode reports which interface dominates.
    assert base_result.fail_mode in {"EBC", "TGO"}
    if base_result.fail_index_EBC >= base_result.fail_index_TGO:
        assert base_result.fail_mode == "EBC"
    else:
        assert base_result.fail_mode == "TGO"


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
