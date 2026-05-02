"""Scale 2 output aggregator (spec §5.6).

STATUS: stub — collects {kappa(T), alpha(T), D_O(T), C_ij(T)} from Scale 2
sub-modules. Intended interface in TEBC_implementation_spec.md §5.6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scale2Outputs:
    kappa_T: np.ndarray
    alpha_T: np.ndarray
    D_O_T:   np.ndarray
    C_ij_T:  np.ndarray


def collect(*args, **kwargs) -> Scale2Outputs:
    raise NotImplementedError(
        "Scale 2 output aggregator not yet implemented. See spec §5.6.",
    )
