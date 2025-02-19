"""
Gjonaj version of Takizuka and Abe related functions (see conference
proceeding at doi: 10.18429/JACoW-FEL2022-WEP14. Include some numba
JIT-compiled code to determine various parameters that appear in the
collisions computations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba

from scipy.constants import c, epsilon_0

if TYPE_CHECKING:
    import xtrack as xt


# ----- Private Helpers ----- #


@numba.jit
def _delta_variance() -> numba.float64:  # type: ignore
    """
    Implements the variance calculation of <δ^2>,
    from Eq (1) of Gjonaj et al. (2022).
    """
    pass
