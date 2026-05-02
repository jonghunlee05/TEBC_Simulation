"""LAMMPS input generator + runner (spec §5.1).

STATUS: stub. The current Scale 2 pipeline reads MD trajectories from disk;
on-the-fly LAMMPS execution via this module is not yet implemented.
"""

from __future__ import annotations


def write_lammps_input(*args, **kwargs) -> str:
    raise NotImplementedError(
        "LAMMPS input generator not yet implemented. See spec §5.1.",
    )


def run_lammps(*args, **kwargs):
    raise NotImplementedError(
        "LAMMPS runner not yet implemented. See spec §5.1.",
    )
