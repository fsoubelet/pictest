"""
SIRE-inspired functions: numba JIT-compiled code to determine
various parameters that appear in the collisions computations.

Ideally this doesn't live for very long.
"""

from __future__ import annotations

import itertools
import random

from typing import TYPE_CHECKING

import numba
import numpy as np

from pictest._cells import find_index_of_all_particles_in_given_cell

if TYPE_CHECKING:
    import xtrack as xt

from scipy.constants import c

# ----- Cell Scattering Functions ----- #


def scatter_cell_maxcol_sire(
    cell_number: int,
    attributions: np.ndarray,
    toty: float,
    volume: float,
    delta_t: float,
    particles: xt.Particles,
    max_collisions: int,
) -> None:
    """
    Considers a single cell of the meshgrid and applies collisions
    to randomly chosen pairs of particles in the cell. Stops when
    the maximum number of collisions has been reached.

    This draws random pairs with replacement for each collision,
    until we reach the max number of collisions to do in the cell.

    Parameters
    ----------
    cell_number : int
        The identifier of the cell to consider.
    attributions : np.ndarray
        The array of cell attributions for each particle, as given
        by the `attribute_particle_cells_like_sire` function.
    tot_y : float
        The total space of the meshgrid in vertical, in [m].
    volume : float
        The volume of the cell to consider, in [m^3]. All cells
        have the same volume as the meshgrid is uniform.
    delta_t : float
        The time step of the IBS effect application, in [s].
        Should depend on the element length and the particle
        velocity.
    particles : xtrack.Particles
        The particles distribution object to consider and act on.
    max_collisions : int
        The maximum number of collisions to apply in the cell.
    """
    # ----------------------------------------------
    # Get some cell-specific parameters we will need
    cell_particles = find_index_of_all_particles_in_given_cell(cell_number, attributions)
    n_macroparts: int = cell_particles.size  # number of parts in this cell
    cell_particles = list(cell_particles)  # INDICES - need as list for sampling
    # ----------------------------------------------
    # Determine the number of collisions to do in this cell. If there are more
    # max collisions than particles, we do 1 less than particles (if there are 2
    # parts, there will be 1 collision), otherwise we do the user-provided max
    n_collisions = n_macroparts - 1 if max_collisions >= n_macroparts else max_collisions
    if n_collisions == 0:  # don't waste time and risk 0-division error
        return
    weight = particles.weight[0]  # same for all, accounts for real part per macroparticle
    density: float = n_macroparts * weight / volume / n_collisions  # computed like SIRE
    # ----------------------------------------------
    # We collide for a long as we have to
    while n_collisions > 0:
        part1, part2 = random.sample(cell_particles, k=2)  # choose a pair
        # Apply a coulomb collision to this particle pair
        collide_particle_pair_sire(
            idx1=part1,
            idx2=part2,
            toty=toty,
            density=density,
            delta_t=delta_t,
            particles=particles,
        )
        # That's one less collision to do
        n_collisions -= 1


def scatter_cell_oneperpart_sire(
    cell_number: int,
    attributions: np.ndarray,
    toty: float,
    volume: float,
    delta_t: float,
    particles: xt.Particles,
    **kwargs,
) -> None:
    """
    Considers a single cell of the meshgrid and applies collisions
    to randomly chosen pairs of particles in the cell. Stops when
    the maximum number of collisions has been reached.

    This draws random pairs without replacement for each collision,
    and does exactly one collision per particle.

    Parameters
    ----------
    cell_number : int
        The identifier of the cell to consider.
    attributions : np.ndarray
        The array of cell attributions for each particle, as given
        by the `attribute_particle_cells_like_sire` function.
    tot_y : float
        The total space of the meshgrid in vertical, in [m].
    volume : float
        The volume of the cell to consider, in [m^3]. All cells
        have the same volume as the meshgrid is uniform.
    delta_t : float
        The time step of the IBS effect application, in [s].
        Should depend on the element length and the particle
        velocity.
    particles : xtrack.Particles
        The particles distribution object to consider and act on.
    kwargs : dict
        Not used, this is merely so the function
        does not crash if given 'max_collisions'.
    """
    # ----------------------------------------------
    # Get some cell-specific parameters we will need
    cell_particles = find_index_of_all_particles_in_given_cell(cell_number, attributions)
    n_macroparts: int = cell_particles.size  # number of parts in this cell
    cell_particles = list(cell_particles)  # INDICES - need as list for sampling
    random.shuffle(cell_particles)  # for randomness
    # ----------------------------------------------
    # Determine an ensemble of pairs, for 1 collision per particle
    # We just batch 2 by 2 since we shuffled (for randomness) above
    collided_pairs = list(itertools.batched(cell_particles, 2))
    n_collisions = len(collided_pairs)
    if n_collisions == 0:  # don't waste time and risk 0-division error
        return
    density: float = n_macroparts / volume / n_collisions  # computed like SIRE
    # ----------------------------------------------
    # We collide for every single possible pair
    try:  # Apply a Coulomb collisions to particle pairs
        for part1, part2 in collided_pairs:
            collide_particle_pair_sire(
                idx1=part1,
                idx2=part2,
                toty=toty,
                density=density,
                delta_t=delta_t,
                particles=particles,
            )
    except ValueError:  # happens for an odd number of particles (last pair has no 'part2')
        return


# ----- Particle Pair Collision Function ----- #


def collide_particle_pair_sire(
    idx1: int, idx2: int, toty: float, density: float, delta_t: float, particles: xt.Particles
) -> None:
    """
    Apply the Coulomb scattering to particles denoted by 'idx1'
    and 'idx2' in the provided `xtrack.Particles` object.

    Parameters
    ----------
    idx1 : int
        Index of the first particle of the pair.
    idx2 : int
        Index of the second particle of the pair.
    toty : float
        The total space in vertical. I have no idea
        what this one is doing here. From SIRE.
    density : float
        The local density of the grid cell in which
        the particles belong.
    delta_t : float
        The time step of the IBS effect application,
        in [s]. Not sure exactly how this is determined.
        In SIRE we give it as input.
    particles : xt.Particles
        The `xtrack.Particles` object with the particles
        information, to be directly modified.
    """
    # ----------------------------------------------
    # Get the context and some global properties
    beta0 = particles.beta0[idx1]  # same for both
    gamma0 = particles.gamma0[idx1]  # same for both
    r0 = particles.get_classical_particle_radius0()
    # ----------------------------------------------
    # Get the particle properties needed for the collision
    px1, px2 = particles.px[idx1], particles.px[idx2]
    py1, py2 = particles.py[idx1], particles.py[idx2]
    delta1, delta2 = particles.delta[idx1], particles.delta[idx2]
    # ----------------------------------------------
    # Determine the scattering angle, random uniform between 0 and 2pi
    phi = 2 * np.pi * np.random.random()
    # ----------------------------------------------
    # Compute the momentum deltas (compiled code)
    # These are already divided by two so directly to apply!
    # This way more of the operations are done in compiled code
    deltap1cmx, deltap1cmy, deltap1cmz = sire_coulomb_collision_deltas(
        px1, px2, py1, py2, delta1, delta2, phi, toty, density, delta_t, beta0, gamma0, r0
    )
    # ----------------------------------------------
    # Apply the deltas to the particles (add to part1, remove from part2)
    particles.px[idx1] += deltap1cmx
    particles.py[idx1] += deltap1cmy
    particles.delta[idx1] += deltap1cmz
    particles.px[idx2] -= deltap1cmx
    particles.py[idx2] -= deltap1cmy
    particles.delta[idx2] -= deltap1cmz


# ----- Collision Deltas Function ----- #


@numba.jit
def sire_coulomb_collision_deltas(
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
        return (0.0, 0.0, 0.0)
    coulog = np.log(coulomb)
    # ----------------------------------------------
    # These I have no idea about, just copied from SIRE
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
