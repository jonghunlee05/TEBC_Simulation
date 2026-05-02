"""CMAS dissolution + apatite nucleation (spec §6.3).

STATUS: stub — the chemical-attack model coupling CMAS dissolution kinetics
into the phase-field solver is not yet implemented. See spec §6.3 for the
intended free-energy form and reaction-diffusion coupling.
"""

from __future__ import annotations


def cmas_dissolution_rate(*args, **kwargs) -> float:
    raise NotImplementedError(
        "CMAS dissolution kinetics not yet implemented. See spec §6.3.",
    )


def apatite_nucleation_rate(*args, **kwargs) -> float:
    raise NotImplementedError(
        "Apatite nucleation model not yet implemented. See spec §6.3.",
    )
