"""DP-GEN active-learning workflow wrapper (spec §4.2).

STATUS: stub. The DP-GEN driver is not yet implemented.
This file exists so `from tebc.scale1_atomistic import dpgen_workflow`
does not break, and so the spec is traceable to a concrete file.
"""

from __future__ import annotations


def run_dpgen_iteration(*args, **kwargs):
    raise NotImplementedError(
        "DP-GEN active-learning loop is not yet implemented. "
        "See TEBC_implementation_spec.md §4.2 for the intended interface.",
    )
