"""
Takizuka and Abe related functions: numba JIT-compiled code to determine
various parameters that appear in the collisions computations.
"""

from __future__ import annotations

import numba
import numpy as np
import xtrack as xt

from scipy.constants import c, epsilon_0


# ----- Private Helpers for Takizuka and Abe ----- #


@numba.jit
def _compute_phi(
    ux: numba.float64,  # type: ignore
    uy: numba.float64,  # type: ignore
) -> numba.float64:  # type: ignore
    """
    Solves the second system of Eq (2) in Takizuka and Abe to
    determine the value of phi. An important consideration is
    that phi - as an angle - has to be positive. We will then
    try the first root of the equation and if its value is not
    positive we will return the second root instead.

    See my solving notebook for the derivation of the formula.
    The roots' analytical forms were determined with sympy and
    the validity of the written solutions (decomposed assembled
    terms) was verified just the same.

    Parameters
    ----------
    ux : float64
        The horizontal velocity difference of the particles.
    uy : float64
        The vertical velocity difference of the particles.
    uz : float64
        The longitudinal velocity difference of the particles.

    Returns
    -------
    phi : float64
        The value of phi, the angle between ux and ut, in radians.
    """
    # Both roots differ by only the sign in front
    # of the square root term in the atan
    # ----------------------------------------------
    # Define term used several times to avoid recomputing
    sqrt_term = np.sqrt(ux**2 + uy**2)
    # ----------------------------------------------
    # Attempt the first root of the equation
    first_root = -2 * np.arctan((ux - sqrt_term) / uy)
    if first_root >= 0:
        return first_root
    # ----------------------------------------------
    # Otherwise attempt the second root of the equation
    second_root = -2 * np.arctan((ux + sqrt_term) / uy)
    return second_root


@numba.jit
def _compute_theta(
    ux: numba.float64,  # type: ignore
    uy: numba.float64,  # type: ignore
    uz: numba.float64,  # type: ignore
    phi: numba.float64,  # type: ignore
) -> numba.float64:  # type: ignore
    """
    Solves the first system of Eq (2) in Takizuka and Abe to
    determine the value of theta. An important consideration
    is that theta - as an angle - has to be positive. We will
    then try the first root of the equation and if it is not
    positive we will return the second root instead.

    It is assumed that phi has already been solved for with
    the second system of Eq (2) in Takizuka and Abe, since
    it is an input to this function.

    See my solving notebook for the derivation of the formula.
    The roots' analytical forms were determined with sympy and
    the validity of the written solutions (decomposed assembled
    terms) was verified just the same.

    Parameters
    ----------
    ux : float64
        The horizontal velocity difference of the particles.
    uy : float64
        The vertical velocity difference of the particles.
    uz : float64
        The longitudinal velocity difference of the particles.
    phi : float64
        The angle between ux and ut, see figure
    """
    # Both roots differ by only the sign in front
    # of the square root term in the atan
    # ----------------------------------------------
    # Define term used everywhere to avoid recomputing
    tan_phi_2 = np.tan(phi / 2)  # compute only once
    # ----------------------------------------------
    # Define terms one by one first for clarity
    term_one = uz * tan_phi_2**2
    term_two = ux**2 * tan_phi_2**4
    term_three = 2 * ux**2 * tan_phi_2**2
    term_four = 4 * ux * uy * tan_phi_2**3
    term_five = 4 * ux * uy * tan_phi_2
    term_six = 4 * uy**2 * tan_phi_2**2
    term_seven = uz**2 * tan_phi_2**4
    term_eight = 2 * uz**2 * tan_phi_2**2
    # ----------------------------------------------
    # Define some common terms to both roots
    sqrt_term = np.sqrt(
        term_two
        - term_three
        + ux**2
        - term_four
        + term_five
        + term_six
        + term_seven
        + term_eight
        + uz**2
    )
    denominator = -ux * tan_phi_2**2 + ux + 2 * uy * tan_phi_2
    # ----------------------------------------------
    # Attempt the first root of the equation
    first_root = -2 * np.atan((term_one + uz - sqrt_term) / denominator)
    if first_root >= 0:
        return first_root
    # ----------------------------------------------
    # Otherwise attempt the second root of the equation
    second_root = -2 * np.atan((term_one + uz + sqrt_term) / denominator)
    return second_root


@numba.jit
def _draw_PHI() -> numba.float64:  # type: ignore
    """
    Draws a random value for the azimuthal angle PHI,
    first seen in Eq (3) in Takizuka and Abe's paper
    and described in the sentence just below.

    Returns
    -------
    PHI : float64
        A random number between 0 and 2pi.
    """
    return np.random.uniform(0, 2 * np.pi)


