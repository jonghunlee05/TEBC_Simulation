"""Scale 1 output aggregator (spec §4.5).

STATUS: stub — collects {C_ijkl, alpha_ij, E_defect, gamma_surf} from the
Scale 1 sub-modules and packages them into the cross-scale handoff struct.
Intended interface is defined in TEBC_implementation_spec.md §4.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scale1Outputs:
    C_ijkl: np.ndarray
    alpha_ij: np.ndarray
    E_defect: float
    gamma_surf: float


def collect(*args, **kwargs) -> Scale1Outputs:
    raise NotImplementedError(
        "Scale 1 output aggregator not yet implemented. "
        "See TEBC_implementation_spec.md §4.5.",
    )
