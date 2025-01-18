"""
Collisions related utility: momentum delta, particle to particle, cell scattering.
"""

from __future__ import annotations

import itertools
import random

import numba
import numpy as np
import xtrack as xt

from scipy.constants import c

from pictest._cells import find_index_of_all_particles_in_given_cell


def collide_particle_pair(
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
    deltap1cmx, deltap1cmy, deltap1cmz = nb_coulomb_collision_deltas(
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


# This one does random pairs until MAX_COLLISIONS
def scatter_cell_maxcol(
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

    THIS DRAWS RANDOM PAIRS WITH REPLACEMENT FOR EACH COLLISION,
    UNTIL WE REACH THE MAX NUMBER OF COLLISIONS TO DO IN  THE CELL.

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
    num_parts: int = cell_particles.size  # number of parts in this cell
    cell_particles = list(cell_particles)  # need as list for sampling
    # ----------------------------------------------
    # Determine the number of collisions to do in this cell. If there are more
    # max collisions than particles, we do 1 less than particles (if there are 2
    # parts, there will be 1 collision), otherwise we do the user-provided max
    n_collisions = num_parts - 1 if max_collisions >= num_parts else max_collisions
    if n_collisions == 0:  # don't waste time and risk 0-division error
        return
    weight = particles.weight[0]  # same for all, accounts for real part per macroparticle
    density: float = num_parts * weight / volume / n_collisions  # computed like SIRE
    # ----------------------------------------------
    # We collide for a long as we have to
    while n_collisions > 0:
        part1, part2 = random.sample(cell_particles, k=2)  # choose a pair
        # Apply a coulomb collision to this particle pair
        collide_particle_pair(
            idx1=part1, idx2=part2, toty=toty, density=density, delta_t=delta_t, particles=particles
        )
        # That's one less collision to do
        n_collisions -= 1


# This one does one collision per particle
def scatter_cell_onecoll_perpart(
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

    THIS DRAWS RANDOM PAIRS WITHOUT REPLACEMENT, AND
    DOES 1 COLLISION PER PARTICLE.

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
    num_parts: int = cell_particles.size  # number of parts in this cell
    cell_particles = list(cell_particles)  # need as list for sampling
    random.shuffle(cell_particles)  # for randomness
    # ----------------------------------------------
    # Determine an ensemble of pairs, for 1 collision per particle
    # We just batch 2 by 2 since we shuffled (for randomness) above
    collided_pairs = list(itertools.batched(cell_particles, 2))
    n_collisions = len(collided_pairs)
    if n_collisions == 0:  # don't waste time and risk 0-division error
        return
    density: float = num_parts / volume / n_collisions  # computed like SIRE
    # ----------------------------------------------
    # We collide for every single possible pair
    try:  # Apply a Coulomb collisions to particle pairs
        for part1, part2 in collided_pairs:
            collide_particle_pair(
                idx1=part1,
                idx2=part2,
                toty=toty,
                density=density,
                delta_t=delta_t,
                particles=particles,
            )
    except ValueError:  # happens for an odd number of particles (last pair has no 'part2')
        return


# ----- Compiled Functions ----- #


@numba.jit
def nb_takizuka_abe_collision_deltas(
    px1: numba.float64,  # type: ignore
    px2: numba.float64,  # type: ignore
    py1: numba.float64,  # type: ignore
    py2: numba.float64,  # type: ignore
    delta1: numba.float64,  # type: ignore
    delta2: numba.float64,  # type: ignore
    mass0: numba.float64,  # type: ignore
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

    Returns
    -------
    deltas : tuple[float64, float64, float64]
        The momentum deltas in horizontal, vertical
        and longitudinal momenta. These are already the
        amount to apply to particles.
    """
    # TODO: for Eq. (8a) compute the Coulomb log for the whole bunch and THEN use that value for every cell
    # ----------------------------------------------
    # Some variables definitions from T&A
    ux = (
        px1 - px2
    ) / mass0  # from Eq. (1) ux = vx1 - vx2 = (px1 / m) - (px2 / m) = (px1 - px2) / m
    uy = (
        py1 - py2
    ) / mass0  # from Eq. (1) uy = vy1 - vy2 = (py1 / m) - (py2 / m) = (py1 - py2) / m
    uz = (
        delta1 - delta2
    ) / mass0  # from Eq. (1) un = vn1 - vn2 = (delta1 / m) - (delta2 / m) = (delta1 - delta2) / m
    # ----------------------------------------------
    # THEN WE NEED TO SOLVE FOR THETA and PHI in Eq. (2)
    # We know the following two relations
    # ----------------------------------------------
    # NOW WE NEED TO DRAW A DELTA VARIABLE FROM A GAUSSIAN DISTRIBUTION
    # with 0 mean and a variance according to Eq. (8a) (remember that
    # alpha and beta species are the same species of particles here):
    # variance = delta_t * (charge_alpha**2 * charge_beta**2 * n_l * coulog) / (8 * pi * epsilon_0**2 * m**2 * v**3)
    # In determination of pairs we read
    # ----------------------------------------------
    # We want to compute deltaux, deltauy, deltauz from Eq (4.a) of T&A
    # We first need to compute U_T defined below Eq. (4d)
    # U_T =
    pass


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
    try the first root of the equation and if it is negative,
    we will return the second root instead.

    See my solving notebook for the derivation of the formula.

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
    first_root = -2 * np.arctan((ux - np.sqrt(ux**2 + uy**2)) / uy)
    if first_root > 0:
        return first_root
    return -2 * np.arctan((ux + np.sqrt(ux**2 + uy**2)) / uy)

