"""
Collisions related utility: momentum delta, particle to particle, cell scattering.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import xtrack as xt

from scipy.constants import c

from pictest._cells import find_index_of_all_particles_in_given_cell
from pictest._sire import collide_particle_pair_sire
from pictest._takizuka_abe import nb_takizuka_abe_collision_deltas

# ----- Functions to Scatter Entire Cells ----- #


# TODO: THIS IS STILL QUITE SIRE SPECIFIC (toty, volume etc), REWORK LATER
# This one does random pairs until MAX_COLLISIONS
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


# TODO: THIS IS STILL QUITE SIRE SPECIFIC (toty, volume etc), REWORK LATER
# This one does one collision per particle
def scatter_cell_onecoll_perpart_sire(
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
