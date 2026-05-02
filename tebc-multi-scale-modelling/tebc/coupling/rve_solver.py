"""FE² RVE micro-BVP solver (spec §8.2).

STATUS: stub. A nested-FE micromechanical RVE driver is not yet implemented;
analytical homogenization in `tebc.coupling.homogenization` is the current
fallback. See TEBC_implementation_spec.md §8.2 for the intended interface.
"""

from __future__ import annotations


def solve_rve_microbvp(*args, **kwargs):
    raise NotImplementedError(
        "FE² RVE micro-BVP solver not yet implemented. See spec §8.2.",
    )
