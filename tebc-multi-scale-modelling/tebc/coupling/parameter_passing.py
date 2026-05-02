"""Validated cross-scale parameter handoff (spec §8.3).

STATUS: stub — the orchestrator currently passes parameters via a single
`ScaleParameters` dataclass without an explicit unit-validation layer.
This module is the intended home for those checks.
"""

from __future__ import annotations

from typing import Any


def validate_handoff(name: str, value: Any, expected_unit: str) -> Any:
    """Placeholder validator.

    Intended to enforce SI units, finite values, and physical bounds on
    quantities passed between scales. Not yet implemented — see spec §8.3.
    """
    raise NotImplementedError(
        "Cross-scale parameter validation not yet implemented. See spec §8.3.",
    )
