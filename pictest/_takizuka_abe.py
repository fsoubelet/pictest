"""
Takizuka and Abe related functions: numba JIT-compiled code to determine
various parameters that appear in the collisions computations.
"""

from __future__ import annotations

import numba
import numpy as np

from scipy.constants import c, epsilon_0

# ----- Exposed Collision Deltas Function ----- #


@numba.jit
def nb_takizuka_abe_collision_deltas(
    px1: numba.float64,  # type: ignore
    px2: numba.float64,  # type: ignore
    py1: numba.float64,  # type: ignore
    py2: numba.float64,  # type: ignore
    delta1: numba.float64,  # type: ignore
    delta2: numba.float64,  # type: ignore
    q0: numba.float64,  # type: ignore
    mass0: numba.float64,  # type: ignore
    coulog: numba.float64,  # type: ignore
    delta_t: numba.float64,  # type: ignore
) -> tuple[numba.float64, numba.float64, numba.float64]:  # type: ignore
    """
    Compute momentum deltas for a two-article Coulomb collision, trying to stay
    exactly on the formulae from Takizuka and Abe. This is a compiled function
    to be used in the main loop of the PIC simulation.

    Notes
    -----
    In here I will be using the exact variable names of the T&A paper,
    but try to comment as much as possible to explain things.

    Parameters
    ----------
    px1 : float64
        Horizontal momentum of the first particle of the pair.
    px2 : float64
        Horizontal momentum of the second particle of the pair.
    py1 : float64
        Vertical momentum of the first particle of the pair.
    py2 : float64
        Vertical momentum of the second particle of the pair.
    delta1 : float64
        Momentum deviation of the first particle of the pair.
    delta2 : float64
        Momentum deviation of the second particle of the pair.
    q0 : float64
        The charge of the particles in the pair (same species),
        in [e].
    mass0 : float64
        The mass of the particles in the pair (same species), in
        [eV] as `xtrack` gives us.
    coulog : float64
        The Coulomb logarithm for the whole bunch.
    delta_t : float64
        The time scale of the IBS interaction, in [s].

    Returns
    -------
    deltas : tuple[float64, float64, float64]
        The momentum deltas in horizontal, vertical
        and longitudinal momenta. These are already the
        amount to apply to particles.
    """
    # ----------------------------------------------
    # First define the 'u' variables from Eq. (1)
    # We divide by mass because xtrack uses momentum
    _ev_to_J = 1.602176634 * 10**-19  # conversion factor from eV to J
    _eV_to_g = 1e3 * _ev_to_J / c**2  # conversion factor from eV to kg
    mass_g = mass0 *_eV_to_g  # we want mass in [g]
    ux = (px1 - px2) / mass_g  # ux = vx1 - vx2
    uy = (py1 - py2) / mass_g  # uy = vy1 - vy2
    uz = (delta1 - delta2) / mass_g  # uz = vz1 - vz2
    # ----------------------------------------------
    # Now we compute phi, theta and then u from Eq. (2)
    # See relevant function docstrings for information
    phi = _compute_phi(ux, uy)
    theta = _compute_theta(ux, uy, uz, phi)
    u = np.sin(theta) * np.cos(phi) * ux + np.sin(theta) * np.sin(phi) * uy + np.cos(theta) * uz
    # ----------------------------------------------
    # We draw the angle PHI from Eq. (3) (uniform dist [0-2pi])
    # and we determine THETA by dr
    PHI = _draw_PHI()
    # ----------------------------------------------
    # We draw a value for delta according to Eq (8a)
    # and then plug into Eq 
    delta = _draw_delta()  # TODO: implement this function
    THETA = 2 * np.arctan(delta)
    # ----------------------------------------------
    # We compute U_T defined below Eq (4d)
    # TODO

    # ----------------------------------------------
    # Now we compute deltaux, deltauy, deltauz from
    # Eq (4.b), Eq (4.c) and Eq (4.d) respectively.
    # TODO

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


def _draw_delta(
    q0: numba.float64,  # type: ignore
    mass_g: numba.float64,  # type: ignore
    coulog: numba.float64,  # type: ignore
    delta_t: numba.float64,  # type: ignore
    u: numba.float64,  # type: ignore
) -> numba.float64:  # type: ignore
    """
    Draws a random value for the variable delta, which is
    used to later determine the scattering angle THETA. Its
    properties are described in Eq (8a) in Takizuka and Abe's
    paper.

    Parameters
    ----------
    q0 : float64
        The charge of the particles in the pair (same species),
        in [e].
    mass_g : float64
        The mass of the particles in the pair (same species),
        in [g].
    coulog : float64
        The Coulomb logarithm for the whole bunch.
    delta_t : float64
        The time scale of the IBS interaction, in [s].
    u : float64
        The transverse velocity of the particles.

    Returns
    -------
    delta : float64
        A random number from the relevant distribution.
    """
    # ----------------------------------------------
    # We compute the variance as described by Eq. (8a)
    #  
    variance = delta_t * () / (8 * np.pi * epsilon_0**2)
    scale = np.std(variance)
    return np.random.normal(0, scale=scale)
