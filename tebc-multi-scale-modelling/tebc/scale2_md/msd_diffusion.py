"""
Mean Square Displacement → diffusion coefficient → Arrhenius fit.

D_O(T) = D0 * exp(-Ea / k_B T)
MSD: ⟨|Δr(t)|²⟩ = (1/N) Σ_i |r_i(t) - r_i(0)|²
"""

import numpy as np
from scipy.stats import linregress

from tebc.utils import arrhenius_fit


def compute_msd(positions: np.ndarray, species_mask: np.ndarray = None,
                max_lag_fraction: float = 0.5):
    """Compute MSD from trajectory."""
    if species_mask is not None:
        positions = positions[:, species_mask, :]
    n_frames, n_atoms, _ = positions.shape
    n_lag = int(n_frames * max_lag_fraction)
    msd = np.zeros(n_lag)
    for lag in range(1, n_lag):
        disp = positions[lag:] - positions[:-lag]
        msd[lag] = np.mean(disp**2)
    return np.arange(n_lag), msd


def msd_to_diffusivity(t_lag: np.ndarray, msd: np.ndarray,
                        dt_per_frame: float,
                        fit_start_frac: float = 0.1,
                        fit_end_frac:   float = 0.5,
                        dim: int = 3) -> dict:
    """Fit D from linear region of MSD: D = slope / (2 * dim)."""
    n = len(t_lag)
    i0 = int(n * fit_start_frac)
    i1 = int(n * fit_end_frac)
    t_fit   = t_lag[i0:i1] * dt_per_frame
    msd_fit = msd[i0:i1]
    slope, intercept, r, p, stderr = linregress(t_fit, msd_fit)
    D = slope / (2 * dim)
    return {"D": D, "D_std": stderr / (2*dim), "r2": r**2}


def arrhenius_diffusivity(T_list: np.ndarray, D_list: np.ndarray) -> dict:
    """Fit D(T) = D0 * exp(-Ea / k_B T)."""
    import warnings

    from tebc.constants import eV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, Ea_J = arrhenius_fit(T_list, D_list)
    return {"D0": A, "Ea_J": Ea_J, "Ea_eV": Ea_J / eV}
