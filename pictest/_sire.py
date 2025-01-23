"""
SIRE-inspired functions: numba JIT-compiled code to determine
various parameters that appear in the collisions computations.

Ideally this doesn't live for very long.
"""

from __future__ import annotations

import itertools
import random

import numba
import numpy as np
import xtrack as xt

from scipy.constants import c, epsilon_0

# ----- Exposed Collision Deltas Function ----- #


@numba.jit
def nb_coulomb_collision_deltas(
    px1: numba.float64,  # type: ignore
    px2: numba.float64,  # type: ignore
    py1: numba.float64,  # type: ignore
    py2: numba.float64,  # type: ignore
    delta1: numba.float64,  # type: ignore
    delta2: numba.float64,  # type: ignore
    phi: numba.float64,  # type: ignore
    toty: numba.float64,  # type: ignore
    density: numba.float64,  # type: ignore
    delta_t: numba.float64,  # type: ignore
    beta0: numba.float64,  # type: ignore
    gamma0: numba.float64,  # type: ignore
    r0: numba.float64,  # type: ignore
) -> tuple[numba.float64, numba.float64, numba.float64]:  # type: ignore
    """
    Determine and return the momentum deltas to apply to each
    particle in the pair from a Coulomb scattering event, based
    on the particles' momenta and some local properties.

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
    phi : float64
        Polar collision angle, provided so we use the same as SIRE.
    toty : float64
        The total space in vertical. I have no idea
        what this one is doing here.
    density : float64
        The local density of the grid cell in which
        the particles belong.
    delta_t : float64
        The time step of the IBS effect application,
        in [s]. Not sure exactly how this is determined.
    beta0 : float64
        Relativistic beta of the particles.
    gamma0 : float64
        Relativistic gamma of the particles.
    r0 : float64
        Classical particle radius.

    Returns
    -------
    deltas : tuple[float64, float64, float64]
        The momentum deltas in horizontal, vertical
        and longitudinal momenta. These are already the
        amount to apply to particles.
    """
    # ----------------------------------------------
    # We compute the delta_u from Takizuka and Abe (momentum diffs)
    deltapcmx = px1 - px2
    deltapcmy = py1 - py2
    deltapcmz = (delta1 - delta2) / gamma0  # gammar in SIRE here for??
    # ----------------------------------------------
    # We compute the transverse u_T and the u_n from SIRE (it is simply u in Gjonaj)
    # deltapcmn is sqrt(deltapcmx**2 + deltapcmy**2 + deltapcmz**2)
    deltapcmt = np.sqrt(deltapcmx**2 + deltapcmy**2)
    deltapcmn = np.sqrt(deltapcmt**2 + deltapcmz**2)
    # ----------------------------------------------
    # phi = 2 * np.pi * rng.random()  # polar collision angle, random uniform between 0 and 2pi
    betatilde = beta0 * gamma0 * deltapcmn / 2
    coulomb = toty * betatilde**2 / r0
    # ----------------------------------------------
    # Firstly, assert we can get a Coulog otherwise ignore all
    if coulomb <= 1:
        print("Coulomb <= 1, ignoring collision")
        return (0.0, 0.0, 0.0)
    # ----------------------------------------------
    # Firstly, assert we can get a Coulog otherwise ignore all
    coulog = np.log(coulomb)
    one_minus_cos_psi = (
        2 * np.pi * c * density * r0**2 * delta_t * coulog / (gamma0**2 * betatilde**3)
    )
    sin_psi = np.sqrt(2) * np.sqrt(one_minus_cos_psi)
    # ----------------------------------------------
    # Two cases on determining the deltap in each dimension
    if deltapcmt != 0:  # There is non-zero deltapcmt (transverse u_T)
        deltap1cmx = (
            -deltapcmx * one_minus_cos_psi
            + (
                sin_psi * np.cos(phi) * deltapcmx * deltapcmz
                - sin_psi * np.sin(phi) * deltapcmn * deltapcmy
            )
            / deltapcmt
        )
        deltap1cmy = (
            -deltapcmy * one_minus_cos_psi
            + (
                sin_psi * np.cos(phi) * deltapcmy * deltapcmz
                - sin_psi * np.sin(phi) * deltapcmn * deltapcmx
            )
            / deltapcmt
        )
        deltap1cmz = gamma0 * (-deltapcmz * one_minus_cos_psi - sin_psi * np.cos(phi) * deltapcmt)
    else:  # There is zero deltapcmt (transverse u_T), special case
        deltap1cmx = deltapcmn * sin_psi * np.cos(phi)
        deltap1cmy = deltapcmn * sin_psi * np.sin(phi)
        deltap1cmz = gamma0 * (-deltapcmn * one_minus_cos_psi)
    # ----------------------------------------------
    # Return the deltas, values already divided by two
    # (this is directly to be added to particle1, removed from particle2)
    return (deltap1cmx / 2, deltap1cmy / 2, deltap1cmz / 2)


# ----- Private Helpers for SIRE ----- #
