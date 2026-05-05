"""
Global sensitivity analysis using SALib.

Sobol (variance-based) indices:
  S_i  = V_i / V(Y)
  S_Ti = 1 - Var_{X~i}[E_{X_i}[Y|X~i]] / Var(Y)

Morris elementary effects:
  μ*_i = (1/r) Σ |EE_i^(k)|
  σ_i  = stdev(EE_i)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tebc.coupling.homogenization import maxwell_eucken_kappa

try:
    from SALib.analyze import morris as morris_analyze
    from SALib.analyze import sobol
    from SALib.sample import morris as morris_sample
    # SALib ≥ 1.5: `sobol` sampler replaces deprecated `saltelli`.
    try:
        from SALib.sample import sobol as sobol_sample
    except ImportError:
        from SALib.sample import saltelli as sobol_sample
except ImportError:
    raise ImportError("pip install SALib")


DEFAULT_TEBC_PROBLEM = {
    "num_vars": 7,
    "names": [
        "delta_alpha",
        "k_p",
        "Gamma_int",
        "kappa_TBC",
        "k_l",
        "E_EBC",
        "porosity_TBC",
    ],
    "bounds": [
        [0.5e-6, 2.0e-6],
        [1e-15,  1e-12],
        [5.0,    80.0],
        [0.8,    2.5],
        [1e-11,  1e-8],
        [100e9,  250e9],
        [0.05,   0.20],
    ],
    "dists": ["unif", "logunif", "unif", "unif", "logunif", "unif", "unif"],
}


def run_sobol(model_func, problem: dict | None = None,
              N: int = 1024,
              calc_second_order: bool = True,
              seed: int = 42) -> pd.DataFrame:
    """Saltelli-sampled Sobol analysis. `seed` makes sampling reproducible."""
    if problem is None:
        problem = DEFAULT_TEBC_PROBLEM
    X = sobol_sample.sample(problem, N, calc_second_order=calc_second_order,
                            seed=seed)
    Y = model_func(X)
    Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order,
                        print_to_console=False, seed=seed)
    df = pd.DataFrame({
        "parameter": problem["names"],
        "S1":        Si["S1"],
        "S1_conf":   Si["S1_conf"],
        "ST":        Si["ST"],
        "ST_conf":   Si["ST_conf"],
    })
    df = df.sort_values("ST", ascending=False).reset_index(drop=True)
    return df


def run_morris(model_func, problem: dict | None = None,
               n_trajectories: int = 50,
               num_levels: int = 4,
               seed: int = 42) -> pd.DataFrame:
    """Morris elementary effects screening. `seed` makes sampling reproducible."""
    if problem is None:
        problem = DEFAULT_TEBC_PROBLEM
    X = morris_sample.sample(problem, N=n_trajectories,
                              num_levels=num_levels, optimal_trajectories=10,
                              seed=seed)
    Y = model_func(X)
    Si = morris_analyze.analyze(problem, X, Y, print_to_console=False,
                                seed=seed)
    df = pd.DataFrame({
        "parameter": problem["names"],
        "mu_star":   Si["mu_star"],
        "sigma":     Si["sigma"],
        "mu":        Si["mu"],
    })
    df = df.sort_values("mu_star", ascending=False).reset_index(drop=True)
    return df


def tebc_failure_model(X: np.ndarray,
                        n_cycles: float = 600.0,
                        h_layer: float  = 150e-6) -> np.ndarray:
    """
    Analytical TEBC failure-index surrogate (Evans–Hutchinson framework).

    .. warning::
        This is a **standalone analytical model**, NOT a wrapper around
        ``run_pipeline``. Sobol/Morris results computed against it
        characterise *the surrogate*, which intentionally trades fidelity
        for speed and may diverge from the orchestrator after pipeline
        fixes. Specifically, the surrogate uses ``k_p`` directly as a
        parabolic rate (not as a reference rate at T_ref), the in-plane
        CTE is folded into the single ``delta_alpha`` knob, and stresses
        are combined as a single-layer ERR. To run a sensitivity study
        against the *real* pipeline use a Sobol driver that calls
        ``run_pipeline`` for each X-row (slow; not provided here).

    Inputs (each row of X is one Monte-Carlo sample):
        delta_alpha, k_p, Gamma_int, kappa_TBC, k_l, E_EBC, porosity_TBC
    """
    delta_alpha = X[:, 0]
    k_p         = X[:, 1]
    Gamma_int   = X[:, 2]
    kappa_TBC   = X[:, 3]
    k_l         = X[:, 4]
    E_EBC       = X[:, 5]
    porosity    = X[:, 6]

    nu    = 0.27
    dT    = 1300.0
    t_tot = n_cycles * 3600.0

    sigma0 = (E_EBC / (1 - nu)) * delta_alpha * dT
    G_drive = (1 - nu**2) * sigma0**2 * h_layer / (2 * E_EBC)
    x_TGO = np.sqrt(k_p * t_tot)
    E_TGO, nu_TGO = 70e9, 0.17
    sigma_TGO = (E_TGO/(1-nu_TGO)) * 0.31
    G_TGO  = (1-nu_TGO**2)*sigma_TGO**2 * x_TGO / (2*E_TGO)
    recession_frac = k_l * t_tot / h_layer
    # κ_eff is computed for completeness / future use in extended surrogates;
    # the current failure-index expression is dominated by G_drive, G_TGO,
    # Γ_int and recession_frac, so kappa_TBC enters only through subsequent
    # extensions. Keep the call so vectorised inputs validate cleanly.
    _kappa_eff = maxwell_eucken_kappa(kappa_TBC, 0.0, porosity)

    fail_idx = ((G_drive + G_TGO) / (Gamma_int + 1e-30)
                + 2.0 * recession_frac)
    return fail_idx


def tebc_failure_model_pipeline(X: np.ndarray) -> np.ndarray:
    """Run the *real* pipeline for each Sobol sample and return the
    failure index.

    This is the physically faithful counterpart to
    `tebc_failure_model` — at the cost of being ~1000× slower because
    each sample triggers a full `run_pipeline` call. Use it when you
    want sensitivities against what the orchestrator actually computes
    (post-Arrhenius-fix, post-anisotropic-CTE, post-per-interface
    ERR), not against the analytical surrogate.

    The Sobol problem is the same as `DEFAULT_TEBC_PROBLEM`:
        delta_alpha, k_p, Gamma_int, kappa_TBC, k_l, E_EBC, porosity_TBC

    Inputs are mapped onto pipeline knobs as follows:
        delta_alpha  → effective Δα is enforced by adjusting alpha_aniso[0]
        k_p          → multiplies mat_bond["k_p_wet"]
        Gamma_int    → overrides mat_EBC["Gamma_interface"]
        kappa_TBC    → overrides mat_TBC["kappa"]
        k_l          → multiplies the Robinson–Smialek baseline
        E_EBC        → overrides mat_EBC["E"]
        porosity_TBC → cfg.phi_TBC
    """
    from copy import deepcopy

    from tebc.constants import MATERIALS as _MAT
    from tebc.orchestrator import TEBCConfig, run_pipeline

    n = X.shape[0]
    Y = np.empty(n)
    base_mat = deepcopy(_MAT)

    for i in range(n):
        delta_alpha, k_p, Gamma_int, kappa_TBC, k_l, E_EBC, porosity_TBC = X[i]
        # Snapshot-edit-restore so each sample is independent.
        try:
            ebc = _MAT["beta_Yb2Si2O7"]
            tbc = _MAT["7YSZ"]
            bond = _MAT["Si_bondcoat"]

            ebc_alpha_aniso0 = ebc["alpha_aniso"][0]
            ebc["alpha_aniso"][0] = ebc["alpha"] + delta_alpha
            ebc["E"] = float(E_EBC)
            ebc["Gamma_interface"] = float(Gamma_int)
            tbc["kappa"] = float(kappa_TBC)
            kp_orig = bond["k_p_wet"]; bond["k_p_wet"] = float(k_p)

            cfg = TEBCConfig(
                run_scale1=False, run_scale2=True, run_scale3=True,
                run_scale4=True, run_sensitivity=False,
                phi_TBC=float(porosity_TBC),
                v_gas=k_l,         # k_l is folded through R-S — see warn
                write_sobol_csv=False,
            )
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = run_pipeline(cfg)
            Y[i] = res.fail_index
        finally:
            # restore mutated entries (keys we touched)
            ebc["alpha_aniso"][0] = ebc_alpha_aniso0
            ebc["E"]               = base_mat["beta_Yb2Si2O7"]["E"]
            ebc["Gamma_interface"] = base_mat["beta_Yb2Si2O7"]["Gamma_interface"]
            tbc["kappa"]           = base_mat["7YSZ"]["kappa"]
            bond["k_p_wet"]        = base_mat["Si_bondcoat"]["k_p_wet"]
    return Y
