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
