"""Scale 3 output aggregator (spec §6.4).

STATUS: stub — collects {kappa_eff, E_eff, TGO_thickness(t)} from Scale 3
sub-solvers. Intended interface in TEBC_implementation_spec.md §6.4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scale3Outputs:
    kappa_eff:    float
    E_eff:        float
    TGO_thickness_t: np.ndarray


def collect(*args, **kwargs) -> Scale3Outputs:
    raise NotImplementedError(
        "Scale 3 output aggregator not yet implemented. See spec §6.4.",
    )
