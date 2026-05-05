"""
Mean Square Displacement → diffusion coefficient → Arrhenius fit.

D_O(T) = D0 * exp(-Ea / k_B T)
MSD: ⟨|Δr(t)|²⟩ = (1/N) Σ_i |r_i(t) - r_i(0)|²
"""

import numpy as np
from scipy.stats import linregress

from tebc.utils import arrhenius_fit


def compute_msd(positions: np.ndarray,
                species_mask: np.ndarray | None = None,
                max_lag_fraction: float = 0.5,
                remove_com_drift: bool = True,
                framework_mask: np.ndarray | None = None,
                masses: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute MSD ⟨|Δr(τ)|²⟩ from a trajectory.

    Returns the *total* mean square displacement (summed over x, y, z),
    averaged over time origins and atoms — the form Einstein's relation
    ⟨|Δr|²⟩ = 2·d·D·t expects (d = 3 in 3D).

    Parameters
    ----------
    positions : (n_frames, n_atoms, 3) array
    species_mask : optional bool mask selecting the diffusing species
        (e.g. O²⁻ in YSZ).
    max_lag_fraction : fraction of the trajectory to evaluate up to.
    remove_com_drift : if True (default), subtract the COM motion of the
        framework (or all atoms if `framework_mask` is None) from every
        frame before computing the MSD. **Critical** for ionic-conduction
        analysis: without it, any net cell drift is folded into the
        diffusivity. Standard MD analysis does this; the previous
        implementation did not.
    framework_mask : bool mask selecting the framework atoms whose COM
        drift is removed. Defaults to all atoms.
    masses : per-atom mass array (used as weights when computing the
        framework COM). If omitted, equal weights are used.

    Performance
    -----------
    Uses an FFT-based autocorrelation: O(N_frames · log N_frames) per
    atom-component. The previous explicit double loop was O(N_frames²),
    which is unusable for typical MD trajectories (10⁵–10⁶ frames).
    """
    pos = np.asarray(positions, dtype=float)

    if remove_com_drift:
        # Subtract the *framework* COM, not the all-atom COM. If the
        # caller did not specify the framework explicitly, default to the
        # complement of `species_mask` — i.e. everything that is NOT the
        # diffusing species. This avoids subtracting a fraction of the
        # diffusing species' own motion (which would underestimate D).
        if framework_mask is None:
            if species_mask is not None:
                framework_mask = ~np.asarray(species_mask, dtype=bool)
                if not framework_mask.any():
                    framework_mask = None  # fall back to all atoms
        if framework_mask is None:
            framework = pos
            w = masses if masses is not None else None
        else:
            framework = pos[:, framework_mask, :]
            w = masses[framework_mask] if masses is not None else None
        if w is None:
            com = framework.mean(axis=1, keepdims=True)            # (n_frames, 1, 3)
        else:
            w = np.asarray(w, dtype=float)
            com = (framework * w[None, :, None]).sum(axis=1, keepdims=True) / w.sum()
        pos = pos - com

    if species_mask is not None:
        pos = pos[:, species_mask, :]

    n_frames, n_atoms, _ = pos.shape
    n_lag = int(n_frames * max_lag_fraction)

    # FFT autocorrelation per atom-component, then sum over xyz and
    # average over atoms. Reference: nMoldyn (Kneller 1995); see also
    # the textbook recipe in Frenkel & Smit, "Understanding Molecular
    # Simulation", Appendix D.
    msd = _msd_fft(pos, n_lag)
    return np.arange(n_lag), msd


def _msd_fft(pos: np.ndarray, n_lag: int) -> np.ndarray:
    """FFT-based MSD estimator (Frenkel & Smit recipe).

    For a single particle:
        S1(m) = (1/(N-m)) Σ_n (r(n+m) - r(n))²
              = D(m) - 2 S2(m)
        D(m)  = sum of squared displacements at lags m and N-m
        S2(m) = autocorrelation ⟨r(n)·r(n+m)⟩

    This avoids the O(N²) double loop entirely.
    """
    n_frames, n_atoms, _ = pos.shape
    msd = np.zeros(n_lag)

    # Sum-of-squared-positions running from both ends → D(m).
    sq = np.sum(pos ** 2, axis=2)                          # (n_frames, n_atoms)
    sumsq = np.zeros((n_frames + 1, n_atoms))
    sumsq[1:] = np.cumsum(sq, axis=0)

    # Autocorrelation via FFT, per axis, summed.
    fft_len = 2 * n_frames
    acf = np.zeros((n_frames, n_atoms))
    for ax in range(3):
        F = np.fft.fft(pos[:, :, ax], n=fft_len, axis=0)
        psd = F * np.conjugate(F)
        ac = np.fft.ifft(psd, axis=0).real[:n_frames]
        acf += ac

    for m in range(n_lag):
        N = n_frames - m
        D_m = sumsq[n_frames - m, :] + sumsq[n_frames, :] - sumsq[m, :] - sumsq[0, :]
        S2_m = acf[m, :] / N
        per_atom = (D_m / N) - 2.0 * S2_m
        msd[m] = per_atom.mean()
    return msd


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
