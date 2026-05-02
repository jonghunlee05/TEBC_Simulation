"""Tvergaard–Hutchinson cohesive zone model (spec §7.4).

The implementation currently lives in
`tebc.scale4_continuum.damage_mechanics.TVHCohesiveZone`; this module
re-exports it so the public layout matches the spec.
"""

from tebc.scale4_continuum.damage_mechanics import TVHCohesiveZone  # noqa: F401

__all__ = ["TVHCohesiveZone"]
