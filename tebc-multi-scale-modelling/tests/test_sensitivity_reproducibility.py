"""Reproducibility checks for the SALib-based sensitivity pipeline."""

import pytest

pytest.importorskip("SALib")

from tebc.sensitivity.sobol_morris import (  # noqa: E402
    DEFAULT_TEBC_PROBLEM,
    run_sobol,
    tebc_failure_model,
)


def test_default_problem_is_well_formed():
    p = DEFAULT_TEBC_PROBLEM
    assert p["num_vars"] == len(p["names"]) == len(p["bounds"]) == len(p["dists"])
    for lo, hi in p["bounds"]:
        assert lo < hi


def test_sobol_seed_reproducible():
    df1 = run_sobol(tebc_failure_model, N=64, seed=123)
    df2 = run_sobol(tebc_failure_model, N=64, seed=123)
    # parameter ordering and indices must be identical
    assert df1["parameter"].tolist() == df2["parameter"].tolist()
    for col in ("S1", "ST"):
        assert (df1[col].values == df2[col].values).all()


def test_sobol_different_seeds_give_different_samples():
    df_a = run_sobol(tebc_failure_model, N=64, seed=1)
    df_b = run_sobol(tebc_failure_model, N=64, seed=2)
    # almost-certainly the two sample paths differ → ST differs
    assert (df_a["ST"].values != df_b["ST"].values).any()
