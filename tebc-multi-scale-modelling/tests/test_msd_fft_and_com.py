"""FFT-based MSD with COM-drift removal.

Two regression tests:
1. The new estimator agrees with the textbook closed form ⟨|Δr|²⟩ = 6Dτ
   for a 3D Brownian walker.
2. Adding a constant velocity drift to *every* atom (a COM translation)
   does not change the recovered diffusivity when remove_com_drift=True.
   Without removal, that drift would falsely inflate D.
"""

import numpy as np
import pytest

from tebc.scale2_md.msd_diffusion import compute_msd, msd_to_diffusivity


def _brownian(D, n_frames, n_atoms, dt, seed):
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(2.0 * D * dt)
    steps = rng.normal(0.0, sigma, size=(n_frames - 1, n_atoms, 3))
    pos = np.zeros((n_frames, n_atoms, 3))
    pos[1:] = np.cumsum(steps, axis=0)
    return pos


def test_fft_msd_recovers_D():
    D_true, dt = 1.0e-9, 1.0e-3
    n_frames, n_atoms = 4000, 50
    pos = _brownian(D_true, n_frames, n_atoms, dt, seed=0)
    t_idx, msd = compute_msd(pos, remove_com_drift=False)
    res = msd_to_diffusivity(t_idx, msd, dt_per_frame=dt)
    assert abs(res["D"] - D_true) / D_true < 0.15


def test_com_drift_removed():
    """A constant cell-wide velocity must not contaminate D when COM
    drift removal is on."""
    D_true, dt = 1.0e-9, 1.0e-3
    n_frames, n_atoms = 4000, 50
    pos = _brownian(D_true, n_frames, n_atoms, dt, seed=1)

    # Inject a uniform translation (e.g. unphysical drift in NPT MD).
    drift_velocity = np.array([2e-2, -1e-2, 5e-3])             # m/s, big
    t_axis = np.arange(n_frames)[:, None, None] * dt
    pos_drifted = pos + drift_velocity[None, None, :] * t_axis

    # Without COM removal: D is grossly overestimated.
    _, msd_no_rm = compute_msd(pos_drifted, remove_com_drift=False)
    D_no_rm = msd_to_diffusivity(np.arange(len(msd_no_rm)),
                                  msd_no_rm, dt_per_frame=dt)["D"]
    assert D_no_rm > 5.0 * D_true

    # With COM removal: D is recovered.
    _, msd_rm = compute_msd(pos_drifted, remove_com_drift=True)
    D_rm = msd_to_diffusivity(np.arange(len(msd_rm)),
                               msd_rm, dt_per_frame=dt)["D"]
    assert abs(D_rm - D_true) / D_true < 0.15


def test_fft_msd_matches_naive_on_small_trajectory():
    """The new FFT path and the textbook double-loop must give the same
    MSD curve (to within float roundoff) on a small reference set."""
    rng = np.random.default_rng(2)
    n_frames, n_atoms = 200, 8
    pos = rng.normal(0.0, 1.0, size=(n_frames, n_atoms, 3))
    _, msd_fft = compute_msd(pos, remove_com_drift=False,
                              max_lag_fraction=0.5)
    # Reference O(N²) calculation (matches the corrected pre-FFT formula
    # — sum over xyz, mean over origins and atoms).
    n_lag = len(msd_fft)
    msd_ref = np.zeros(n_lag)
    for lag in range(1, n_lag):
        disp = pos[lag:] - pos[:-lag]
        msd_ref[lag] = np.sum(disp ** 2, axis=2).mean()
    np.testing.assert_allclose(msd_fft[1:], msd_ref[1:], rtol=1e-8, atol=1e-10)
