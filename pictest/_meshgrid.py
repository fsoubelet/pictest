"""
Meshgrid related utility: class, limits, creation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xtrack as xt

from xfields.fieldmaps.interpolated import _configure_grid

Range = tuple[float, float]


@dataclass
class MeshGrid:
    x_grid: np.ndarray  # grid points in horizontal only
    y_grid: np.ndarray  # grid points in vertical only
    z_grid: np.ndarray  # grid points in longitudinal only
    XX: np.ndarray  # Meshgrid points in horizontal
    YY: np.ndarray  # Meshgrid points in vertical
    ZZ: np.ndarray  # Meshgrid points in longitudinal

    @property
    def nx(self) -> int:
        """The number of cells in x."""
        return self.x_grid.size - 1

    @property
    def ny(self) -> int:
        """The number of cells in y."""
        return self.y_grid.size - 1

    @property
    def nz(self) -> int:
        """The number of cells in z."""
        return self.z_grid.size - 1

    @property
    def dx(self) -> float:
        """The cell size in x, in [m]."""
        return self.x_grid[1] - self.x_grid[0]

    @property
    def dy(self) -> float:
        """The cell size in y, in [m]."""
        return self.y_grid[1] - self.y_grid[0]

    @property
    def dz(self) -> float:
        """The cell size in z, in [m]."""
        return self.z_grid[1] - self.z_grid[0]

    @property
    def cell_volume(self) -> float:
        """The volume of a single cell in the grid, in [m^3]."""
        return self.dx * self.dy * self.dz

    @property
    def totx(self) -> float:
        """The total space in x."""
        return self.x_grid[-1] - self.x_grid[0]

    @property
    def toty(self) -> float:
        """The total space in y."""
        return self.y_grid[-1] - self.y_grid[0]

    @property
    def totz(self) -> float:
        """The total space in z."""
        return self.z_grid[-1] - self.z_grid[0]

    @property
    def ncells_transverse(self) -> int:
        """The total number of cells in transverses planes."""
        return self.nx * self.ny

    @property
    def ncells_total(self) -> int:
        """The total number of cells in the grid."""
        return self.nx * self.ny * self.nz


def mesh_limits(particles: xt.Particles, use_delta: bool = False) -> tuple[Range, Range, Range]:
    """
    Give back (min, max) pairs for the horizontal, vertical and
    longitudinal limits of the meshgrid in a way that encapsulates
    all particles in the distribution. It is done like SIRE, a.k.a
    by taking a small margin from the extrema of the particle
    coordinates.

    Parameters
    ----------
    particles : xtrack.Particles
        The particles distribution to get the limits from.
    use_delta : bool, optional
        Whether to use the delta (momentum) values for the
        longitudinal limits. Defaults to `False`.

    Returns
    -------
    tuple[Range, Range, Range]
        The (min, max) pairs for each dimension.
    """
    xmin = particles.x.min() * 1.0001
    xmax = particles.x.max() * 1.0001
    ymin = particles.y.min() * 1.0001
    ymax = particles.y.max() * 1.0001
    zmin = particles.delta.min() * 1.0001 if use_delta is True else particles.zeta.min() * 1.0001
    zmax = particles.delta.max() * 1.0001 if use_delta is True else particles.zeta.max() * 1.0001
    return ((float(xmin), float(xmax)), (float(ymin), float(ymax)), (float(zmin), float(zmax)))


def create_mesh(
    x_space: np.ndarray | None = None,  # array of grid points in horizontal
    y_space: np.ndarray | None = None,  # array of grid points in vertical
    z_space: np.ndarray | None = None,  # array of grid points in longitudinal
    nx: int | None = None,  # ncells in horizontal
    ny: int | None = None,  # ncells in vertical
    nz: int | None = None,  # ncells in longitudinal
    dx: float | None = None,  # cell size in horizontal
    dy: float | None = None,  # cell size in vertical
    dz: float | None = None,  # cell size in longitudinal
    x_range: Range | None = None,  # (min, max) in horizontal
    y_range: Range | None = None,  # (min, max) in vertical
    z_range: Range | None = None,  # (min, max) in longitudinal
) -> MeshGrid:
    """
    Create the grid mesh from the given parameters.

    Parameters
    ----------
    x_space : np.ndarray, optional
        Array of grid points in x.
    y_space : np.ndarray, optional
        Array of grid points in y.
    z_space : np.ndarray, optional
        Array of grid points in z.
    nx : int, optional
        Number of cells in x.
    ny : int, optional
        Number of cells in y.
    nz : int, optional
        Number of cells in z.
    dx : float, optional
        Cell size in x.
    dy : float, optional
        Cell size in y.
    dz : float, optional
        Cell size in z.
    x_range : tuple[float, float], optional
        Range of x values as a (min, max) pair.
    y_range : tuple[float, float], optional
        Range of y values as a (min, max) pair.
    z_range : tuple[float, float], optional
        Range of z values as a (min, max) pair.

    Returns
    -------
    mesh : MeshGrid
        The meshgrid object with the grid points and meshgrid points.
    """
    # Giving n[xyz]+1 as it's taken as the number of points that
    # delimit the grid, so there is 1 less cell in this dimension
    x_space = _configure_grid("x", x_space, dx, x_range, nx + 1)
    y_space = _configure_grid("y", y_space, dy, y_range, ny + 1)
    z_space = _configure_grid("z", z_space, dz, z_range, nz + 1)
    return MeshGrid(x_space, y_space, z_space, *np.meshgrid(x_space, y_space, z_space))
