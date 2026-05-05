"""
Green-Kubo thermal conductivity from MD trajectory.

κ_αβ = (V / k_B T²) ∫₀^∞ ⟨J_α(0) J_β(t)⟩ dt
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

from tebc.constants import k_B


def compute_hcacf(J: np.ndarray, dt: float,
                   max_lag_steps: int | None = None,
                   unbiased: bool = False):
    """Compute heat current autocorrelation function.

    Parameters
    ----------
    J : (n_steps, 3) or (n_steps,) array
        Heat current in W/m² (or any consistent units).
    dt : float
        Timestep [s].
    max_lag_steps : int | None
        Truncate the returned ACF at this lag.
    unbiased : bool
        If False (default), normalise by `n` — the *biased* estimator,
        standard in Green-Kubo because it suppresses noise at long lags.
        If True, normalise by `n − lag` for an *unbiased* estimator;
        gives correct C(τ) magnitude for short trajectories but amplifies
        noise at large τ. The integration window passed to
        `integrate_hcacf` should match the noise budget of either choice.
    """
    if J.ndim == 2:
        HCACF = sum(
            np.correlate(J[:,i], J[:,i], mode='full') for i in range(3)
        ) / 3.0
    else:
        HCACF = np.correlate(J, J, mode='full')

    n = len(J)
    mid = len(HCACF) // 2
    HCACF = HCACF[mid:]
    if unbiased:
        # Divide each lag by (n - lag); avoid division by zero at lag=n.
        denom = np.arange(n, 0, -1, dtype=float)
        HCACF = HCACF / denom
    else:
        HCACF = HCACF / n

    if max_lag_steps is not None:
        HCACF = HCACF[:max_lag_steps]
    t_lag = np.arange(len(HCACF)) * dt
    return t_lag, HCACF


def integrate_hcacf(t_lag: np.ndarray, HCACF: np.ndarray,
                     V: float, T: float) -> np.ndarray:
    """κ(t) = (V / k_B T²) ∫₀^t C(t') dt'"""
    prefactor = V / (k_B * T**2)
    kappa_t = prefactor * cumulative_trapezoid(HCACF, t_lag, initial=0)
    return kappa_t


def plateau_estimate(kappa_t: np.ndarray, t_lag: np.ndarray,
                      t_plateau_start: float = None) -> dict:
    """Estimate plateau κ as mean over [t_plateau_start, t_max]."""
    t_max = t_lag[-1]
    if t_plateau_start is None:
        t_plateau_start = 0.5 * t_max
    mask = t_lag >= t_plateau_start
    kappa_plateau = kappa_t[mask]
    kappa_mean = np.mean(kappa_plateau)
    kappa_std  = np.std(kappa_plateau)
    return {
        "kappa": kappa_mean,
        "kappa_std": kappa_std,
        "converged": (kappa_std / (abs(kappa_mean) + 1e-30)) < 0.15,
    }


def kappa_anisotropic(J_xyz: np.ndarray, dt: float,
                       V: float, T: float) -> np.ndarray:
    """Compute full 3×3 κ tensor."""
    kappa = np.zeros((3, 3))
    prefactor = V / (k_B * T**2)
    n = len(J_xyz)
    # NumPy 2.0 removed np.trapz in favour of np.trapezoid; fall back if
    # only the legacy name is present (NumPy < 2).
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    for a in range(3):
        for b in range(3):
            c = np.correlate(J_xyz[:, a], J_xyz[:, b], mode='full')
            c = c[n-1:] / n
            kappa[a, b] = prefactor * trapezoid(c, dx=dt)
    return kappa
