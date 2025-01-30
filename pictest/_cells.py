"""
Cells related utility: attributions, grouping, etc.
"""

from __future__ import annotations

import numba
import numpy as np
import xtrack as xt

from pictest._meshgrid import MeshGrid


def attribute_particle_cells(
    particles: xt.Particles, meshgrid: MeshGrid, use_delta: bool = False
) -> np.ndarray:
    """
    Given a particles object and a corresponding meshgrid, attributes to
    each particle a cell number as SIRE does. This is done by finding the
    cell in which the particle is located in the meshgrid (simplified).

    Parameters
    ----------
    particles : xtrack.Particles
        The particles distribution to attribute cells to.
    meshgrid : MeshGrid
        The meshgrid object to use for the cell attribution.
        It is used to determine some properties passed to the
        numba assignment function.
    use_delta : bool, optional
        Whether to use the delta (momentum) values for the
        longitudinal coordinates. Defaults to False.

    Returns
    -------
    part_cells : np.ndarray[int64]
        Array with the cell number for each particle. The cell number
        for particle 'n' (with coordinates accessed from the particles
        object with `particles.x[n]` etc) is given at index 'n' in the
        returned array, a.k.a `part_cells[n]`.
    """
    minx, miny, minz = meshgrid.x_grid.min(), meshgrid.y_grid.min(), meshgrid.z_grid.min()
    part_cells = np.zeros_like(particles.x)
    nb_attribute_cells(
        attributions=part_cells,
        x=particles.x,
        y=particles.y,
        z=particles.delta if use_delta is True else particles.zeta,
        minx,
        miny,
        minz,
        meshgrid.dx,
        meshgrid.dy,
        meshgrid.dz,
        meshgrid.nx,
        meshgrid.ny,
        meshgrid.nz,
    )
    return part_cells.astype(int)  # ensure integer


# ----- Compiled Functions ----- #


@numba.jit(parallel=True)
def nb_attribute_cells(
    attributions: numba.float64[:],  # type: ignore
    x: numba.float64[:],  # type: ignore
    y: numba.float64[:],  # type: ignore
    z: numba.float64[:],  # type: ignore
    minx: numba.float64,  # type: ignore
    miny: numba.float64,  # type: ignore
    minz: numba.float64,  # type: ignore
    deltax: numba.float64,  # type: ignore
    deltay: numba.float64,  # type: ignore
    deltaz: numba.float64,  # type: ignore
    ncellsx: numba.int64,  # type: ignore
    ncellsy: numba.int64,  # type: ignore
    ncellsz: numba.int64,  # type: ignore
):
    """
    Attributes each particle to a cell (number), placing the results in
    the 'attributions' array parameter. The value at index n in this
    array hold the cell number for particle n. Attribution is determined
    based on the particle coordinates and the properties of the meshgrid.

    Parameters
    ----------
    attributions : array64
        An array with a value corresponding to the attributed
        cell, for each individual particle.
    x : array64
        Array with the x coordinates of all particles, as given
        by the particles object.
    y : array64
        Array with the y coordinates of all particles, as given
        by the particles object.
    z : array64
        Array with the chosen longitudinal coordinates of all
        particles, as given by the particles object. In SIRE
        the longitudinal 'momentum'-equivalent 'deltasp' is
        used, but we can also use 'zeta' (z). Chosen by what
        is provided to this function at call time.
    minx : float64
        The min value in x of the meshgrid.
    miny : float64
        The min value in y of the meshgrid.
    minz : float64
        The min value in zeta of the meshgrid.
    deltax : float64
        The (uniform) delta in x between edges of a given cell.
    deltay : float64
        The (uniform) delta in y between edges of a given cell.
    deltaz : float64
        The (uniform) delta in zeta between edges of a given cell.
    ncellsx : int64
        The number of cells in x of the meshgrid.
    ncellsy : int64
        The number of cells in y of the meshgrid.
    ncellsz : int64
        The number of cells in zeta of the meshgrid.
    """
    # This is the SIRE-like way. Notice that we add ints so we
    # might get the same cell (int) for particles that are far
    # away because for instance (4 + 2 + 6) = (8 + 2 + 0) = 10
    # fmt: off
    for part_index in numba.prange(x.size):
        attributions[part_index] = (
            int(np.floor((x[part_index] - minx) / deltax)) * ncellsx  # integer of the cell in horitontal
            + int(np.floor((y[part_index] - miny) / deltay)) * ncellsy  # integer of the cell in vertical
            + int(np.floor((z[part_index] - minz) / deltaz)) * ncellsz  # integer of the cell in longitudinal
        )
    # fmt: on


@numba.jit
def find_index_of_all_particles_in_given_cell(
    cell_number: numba.int64,  # type: ignore
    cell_attributions: numba.float64[:],  # type: ignore
) -> numba.float64[:]:  # type: ignore
    """
    For a given particle (index), return the particle indices
    of all other particles located in the same grid cell.

    Parameters
    ----------
    cell_number : int64
        The number (associated identifier) of the given cell.
    cell_attributions : array64
        The array of cell attributions for each particle, as given
        by the `attribute_particle_cells_like_sire` function.

    Returns
    -------
    indices : array64
        The indices of all particles in this specific cell as the
        particle at `part_index`.
    """
    return np.argwhere(cell_attributions == cell_number).flatten()


@numba.jit
def find_index_of_all_particles_in_same_cell(
    part_index: numba.int64,  # type: ignore
    cell_attributions: numba.float64[:],  # type: ignore
) -> numba.float64[:]:  # type: ignore
    """
    For a given particle (index), return the particle indices
    of all other particles located in the same grid cell.

    Parameters
    ----------
    part_index : int64
        The index of the particle to find "cell neighbors" for.
    cell_attributions : array64
        The array of cell attributions for each particle, as given
        by the `attribute_particle_cells_like_sire` function.

    Returns
    -------
    indices : array64
        The indices of all particles in the same cell as the
        particle at `part_index`.
    """
    cell_for_particle = cell_attributions[part_index]
    return find_index_of_all_particles_in_given_cell(cell_for_particle, cell_attributions)


@numba.jit
def find_number_of_particles_in_given_cell(
    cell_number: numba.int64,  # type: ignore
    cell_attributions: numba.float64[:],  # type: ignore
) -> numba.int64:  # type: ignore
    """
    For a given particle (index), return the particle indices
    of all other particles located in the same grid cell.

    Parameters
    ----------
    cell_number : int64
        The number (associated identifier) of the given cell.
    cell_attributions : array64
        The array of cell attributions for each particle, as given
        by the `attribute_particle_cells_like_sire` function.

    Returns
    -------
    amount : int64
        The number of particles in this given cell.
    """
    return find_index_of_all_particles_in_given_cell(cell_number, cell_attributions).size
