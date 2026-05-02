"""
Phonon calculations and Quasi-Harmonic Approximation (QHA).

QHA Helmholtz free energy:
  F(V,T) = E0(V) + Σ_{q,s} ½ℏω_{q,s}(V)
                 + k_B T Σ_{q,s} ln[1 - exp(-ℏω_{q,s}(V)/k_BT)]
"""

from __future__ import annotations

import numpy as np

from tebc.constants import hbar, k_B
from tebc.utils import fit_eos, mode_heat_capacity


def compute_free_energy(omega_qpts: np.ndarray, weights: np.ndarray,
                         E0: float, T: float) -> float:
    """F(V,T) = E0 + Σ_{q,s} w_q [½ℏω + k_BT ln(1-exp(-ℏω/k_BT))]"""
    kBT = k_B * T
    hw  = hbar * omega_qpts
    F_ZPE = 0.5 * (weights[:, None] * hw).sum()
    x  = hw / kBT
    # ln(1 - exp(-x)) is asymptotic to -exp(-x) for large x and to ln(x) for
    # small x.  We use a piecewise-stable form to avoid log-of-zero and
    # cancellation: at the high-x end the contribution is essentially zero
    # (~ -exp(-x)), at the low-x end we evaluate np.log1p(-exp(-x)) which
    # is well conditioned because exp(-x) ≪ 1 by then anyway.
    ln_term = np.where(x > 50.0,
                        -np.exp(-np.minimum(x, 700.0)),
                        np.log1p(-np.exp(-np.clip(x, 1e-30, 50.0))))
    F_vib   = kBT * (weights[:, None] * ln_term).sum()
    return E0 + F_ZPE + F_vib


def qha_cte(V_list: np.ndarray, T_list: np.ndarray,
             omega_list: list, weights: np.ndarray,
             E_list: np.ndarray) -> dict:
    """Perform QHA: fit E(V), compute V_eq(T), CTE α_V(T), Grüneisen γ."""
    n_V, n_T = len(V_list), len(T_list)
    V_eq    = np.zeros(n_T)
    B_T     = np.zeros(n_T)

    for i, T in enumerate(T_list):
        F_V = np.array([
            compute_free_energy(omega_list[j], weights, E_list[j], T)
            for j in range(n_V)
        ])
        eos = fit_eos(V_list, F_V)
        V_eq[i] = eos["V0"]
        B_T[i]  = eos["B0"]

    dV_dT = np.gradient(V_eq, T_list)
    alpha_V = dV_dT / V_eq

    mid = n_V // 2
    omega_ref = omega_list[mid]
    dln_omega = np.zeros_like(omega_ref)
    if n_V >= 3:
        ln_V = np.log(V_list)
        dln_omega = (np.log(omega_list[-1]+1e-30) -
                     np.log(omega_list[0]+1e-30)) / (ln_V[-1] - ln_V[0])
    gamma_mode = -dln_omega

    T_ref = T_list[len(T_list)//2]
    C_qs  = mode_heat_capacity(omega_ref, T_ref)
    C_tot = (weights[:, None] * C_qs).sum()
    gamma_avg = (weights[:, None] * C_qs * gamma_mode).sum() / (C_tot + 1e-30)

    return {
        "V_eq":    V_eq,
        "T_list":  T_list,
        "alpha_V": alpha_V,
        "alpha_linear": alpha_V / 3.0,
        "B_T":     B_T,
        "gamma_gruneisen": gamma_avg,
    }


def gruneisen_cte_relation(gamma: float, Cv: float, V: float, B: float) -> float:
    """α_V = γ C_V / (B V)  [Grüneisen identity]"""
    return gamma * Cv / (B * V)
