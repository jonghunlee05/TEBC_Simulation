"""Paralinear oxidation + Robinson–Smialek recession (spec §7.5).

This module re-exports the kinetics implementations that currently live in
`tebc.scale3_mesoscale.tgo_kinetics` so the public layout matches the spec.
The functions themselves are unchanged.
"""

from tebc.scale3_mesoscale.tgo_kinetics import (  # noqa: F401
    deal_grove_thickness,
    robinson_smialek_recession,
    solve_paralinear,
)

__all__ = [
    "deal_grove_thickness",
    "robinson_smialek_recession",
    "solve_paralinear",
]
