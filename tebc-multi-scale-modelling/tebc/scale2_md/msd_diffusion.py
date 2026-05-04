"""
Mean Square Displacement → diffusion coefficient → Arrhenius fit.

D_O(T) = D0 * exp(-Ea / k_B T)
MSD: ⟨|Δr(t)|²⟩ = (1/N) Σ_i |r_i(t) - r_i(0)|²
"""

import numpy as np
from scipy.stats import linregress

from tebc.utils import arrhenius_fit


def compute_msd(positions: np.ndarray, species_mask: np.ndarray | None = None,
                max_lag_fraction: float = 0.5):
    """Compute MSD ⟨|Δr(τ)|²⟩ from a trajectory.

    Returns the *total* mean square displacement (summed over x, y, z),
    averaged over time origins and atoms — the form Einstein's relation
    ⟨|Δr|²⟩ = 2·d·D·t expects, with d = 3 in 3D.

    The earlier implementation used `np.mean(disp**2)`, which averages
    over xyz as well as atoms/time-origins and therefore returned MSD/3.
    Chained into `msd_to_diffusivity` (which divides by `2*dim = 6`
    assuming a *full* MSD) it gave D ÷ 3.

    `positions` shape: (n_frames, n_atoms, 3).
    """
    if species_mask is not None:
        positions = positions[:, species_mask, :]
    n_frames = positions.shape[0]
    n_lag = int(n_frames * max_lag_fraction)
    msd = np.zeros(n_lag)
    for lag in range(1, n_lag):
        disp = positions[lag:] - positions[:-lag]              # (n_frames-lag, n_atoms, 3)
        sq_disp = np.sum(disp ** 2, axis=2)                    # |Δr|² per (origin, atom)
        msd[lag] = sq_disp.mean()                              # ⟨|Δr|²⟩
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
