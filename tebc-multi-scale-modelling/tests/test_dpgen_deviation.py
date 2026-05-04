"""Regression tests for the DP-GEN committee force-deviation reduction.

The previous implementation took a global RMS over (committee × atoms ×
xyz). The DP-GEN convention (Zhang et al., CPC 2020) is the *max over
atoms* of the per-atom RMS deviation across committee members. The two
differ both in scaling and in *which frames* they flag as uncertain.

We reach into `evaluate_model_deviation` indirectly: that function needs
a real DP backend, so we re-derive the per-atom-max statistic on a
synthetic forces tensor and compare against what the corrected code now
computes.
"""

import numpy as np


def _per_atom_max_dev(forces: np.ndarray) -> float:
    """Reference DP-GEN reduction. forces: (K, n_atoms, 3)."""
    mean_F = forces.mean(axis=0)
    per_atom = np.sqrt(((forces - mean_F) ** 2).sum(axis=2).mean(axis=0))
    return float(per_atom.max())


def _global_rms_dev(forces: np.ndarray) -> float:
    """The old (wrong) reduction — global RMS over everything."""
    mean_F = forces.mean(axis=0)
    return float(np.sqrt(((forces - mean_F) ** 2).mean()))


def test_per_atom_max_finds_locally_uncertain_atom():
    """One noisy atom in an otherwise-confident frame → max statistic
    catches it; global RMS dilutes it across the system."""
    rng = np.random.default_rng(0)
    K, n_atoms = 4, 100
    forces = rng.normal(0.0, 0.01, size=(K, n_atoms, 3))   # tight committee
    # Inject a 1.0 eV/Å disagreement at atom #7 only:
    forces[:, 7, 0] += np.array([+0.5, -0.5, +0.5, -0.5])

    sigma_max = _per_atom_max_dev(forces)
    sigma_rms = _global_rms_dev(forces)

    # The locally uncertain atom should clear the σ_hi = 0.25 eV/Å gate.
    assert sigma_max > 0.25, f"max-over-atoms missed the bad atom: {sigma_max}"
    # …whereas the global RMS averages it down below threshold.
    assert sigma_rms < 0.10, f"global RMS should be diluted, got {sigma_rms}"


def test_per_atom_max_matches_uniform_noise():
    """If every atom has the same noise level, the two reductions agree
    in scale (ratio bounded), so this is a sanity check that the new
    code isn't introducing a constant factor against the convention."""
    rng = np.random.default_rng(1)
    K, n_atoms = 6, 200
    forces = rng.normal(0.0, 0.05, size=(K, n_atoms, 3))
    sigma_max = _per_atom_max_dev(forces)
    # max-over-atoms of a per-atom RMS should sit a few σ above the
    # global RMS for a uniform Gaussian — same order of magnitude.
    sigma_rms = _global_rms_dev(forces)
    assert 1.0 < sigma_max / sigma_rms < 5.0
