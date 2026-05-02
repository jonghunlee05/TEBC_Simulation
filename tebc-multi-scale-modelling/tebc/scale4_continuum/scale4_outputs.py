"""Scale 4 output aggregator (spec §7.6).

STATUS: stub — collects {sigma(x,t), T(x,t), D(x,t), TGO(t)} from Scale 4
sub-solvers. Intended interface in TEBC_implementation_spec.md §7.6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scale4Outputs:
    sigma_xt: np.ndarray
    T_xt:     np.ndarray
    D_xt:     np.ndarray
    TGO_t:    np.ndarray


def collect(*args, **kwargs) -> Scale4Outputs:
    raise NotImplementedError(
        "Scale 4 output aggregator not yet implemented. See spec §7.6.",
    )
