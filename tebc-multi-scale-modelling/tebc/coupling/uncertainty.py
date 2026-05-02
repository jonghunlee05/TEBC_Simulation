"""Surrogates and uncertainty quantification (spec §8.4).

STATUS: stub. Gaussian-process surrogates, polynomial-chaos expansion, and
Bayesian UQ are not yet implemented. See TEBC_implementation_spec.md §8.4
for the intended interface.
"""

from __future__ import annotations


def fit_gp_surrogate(*args, **kwargs):
    raise NotImplementedError(
        "GP surrogate fitting not yet implemented. See spec §8.4.",
    )


def fit_pce_surrogate(*args, **kwargs):
    raise NotImplementedError(
        "Polynomial-chaos expansion not yet implemented. See spec §8.4.",
    )


def bayesian_inference(*args, **kwargs):
    raise NotImplementedError(
        "Bayesian inference loop not yet implemented. See spec §8.4.",
    )
