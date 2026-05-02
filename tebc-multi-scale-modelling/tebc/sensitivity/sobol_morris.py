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

try:
    from SALib.analyze import sobol, morris as morris_analyze
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


def run_sobol(model_func, problem: dict = None,
              N: int = 1024,
              calc_second_order: bool = True) -> pd.DataFrame:
    """Saltelli-sampled Sobol analysis."""
    if problem is None:
        problem = DEFAULT_TEBC_PROBLEM
    X = sobol_sample.sample(problem, N, calc_second_order=calc_second_order)
    Y = model_func(X)
    Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order,
                        print_to_console=False)
    df = pd.DataFrame({
        "parameter": problem["names"],
        "S1":        Si["S1"],
        "S1_conf":   Si["S1_conf"],
        "ST":        Si["ST"],
        "ST_conf":   Si["ST_conf"],
    })
    df = df.sort_values("ST", ascending=False).reset_index(drop=True)
    return df


def run_morris(model_func, problem: dict = None,
               n_trajectories: int = 50,
               num_levels: int = 4) -> pd.DataFrame:
    """Morris elementary effects screening."""
    if problem is None:
        problem = DEFAULT_TEBC_PROBLEM
    X = morris_sample.sample(problem, N=n_trajectories,
                              num_levels=num_levels, optimal_trajectories=10)
    Y = model_func(X)
    Si = morris_analyze.analyze(problem, X, Y, print_to_console=False)
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
    Analytical TEBC failure index (surrogate for full FEA).
    Based on Evans-Hutchinson framework.
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
    kappa_eff = maxwell_eucken_kappa_simple(kappa_TBC, 0.0, porosity)

    fail_idx = ((G_drive + G_TGO) / (Gamma_int + 1e-30)
                + 2.0 * recession_frac)
    return fail_idx


def maxwell_eucken_kappa_simple(ks, kp, phi):
    """Inline Maxwell-Eucken for sensitivity model."""
    num = 2*ks + kp - 2*phi*(ks - kp)
    den = 2*ks + kp +   phi*(ks - kp)
    return ks * num / (den + 1e-30)
