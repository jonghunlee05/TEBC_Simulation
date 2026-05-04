"""End-to-end MSD → diffusivity round-trip.

Regression test for the bug where `compute_msd` returned ⟨|Δr|²⟩/3
(np.mean over the xyz axis) while `msd_to_diffusivity` divided by
`2·dim = 6` expecting the *full* MSD — net result was D recovered
3× too small when the two functions were chained.
"""

import numpy as np

from tebc.scale2_md.msd_diffusion import compute_msd, msd_to_diffusivity


def _ballistic_drift_trajectory(D_target: float, n_frames: int,
                                 n_atoms: int, dt: float, seed: int = 0):
    """Build a fake trajectory whose MSD = 6·D·t exactly (Brownian).

    Each atom does an independent isotropic random walk with step
    variance σ² = 2·D·dt per spatial component. Then ⟨|Δr(τ)|²⟩ = 6·D·τ
    in 3D for any lag τ.
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(2.0 * D_target * dt)
    steps = rng.normal(0.0, sigma, size=(n_frames - 1, n_atoms, 3))
    positions = np.zeros((n_frames, n_atoms, 3))
    positions[1:] = np.cumsum(steps, axis=0)
    return positions


def test_compute_msd_recovers_6_D_t():
    """⟨|Δr(τ)|²⟩ should equal 6·D·τ for a 3D Brownian walker."""
    D_true = 1.0e-9
    dt = 1.0e-3
    n_frames, n_atoms = 4000, 200
    pos = _ballistic_drift_trajectory(D_true, n_frames, n_atoms, dt)
    t_idx, msd = compute_msd(pos)
    # Fit the linear MSD region only (skip τ=0 and the noisy tail).
    mid = slice(int(0.1 * n_frames), int(0.5 * n_frames))
    t = t_idx[mid] * dt
    slope, _ = np.polyfit(t, msd[mid], 1)
    # slope ≈ 6·D — large statistical sample → 5% tolerance is plenty.
    assert slope == np.float64(slope)  # not NaN
    assert abs(slope - 6.0 * D_true) / (6.0 * D_true) < 0.05


def test_msd_pipeline_recovers_D():
    """Full chain compute_msd → msd_to_diffusivity should return D, not D/3."""
    D_true = 5.0e-10
    dt = 1.0e-3
    n_frames, n_atoms = 4000, 200
    pos = _ballistic_drift_trajectory(D_true, n_frames, n_atoms, dt, seed=1)
    t_idx, msd = compute_msd(pos)
    result = msd_to_diffusivity(t_idx, msd, dt_per_frame=dt)
    assert abs(result["D"] - D_true) / D_true < 0.05
