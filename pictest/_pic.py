"""
PIC IBS functionality: prototype element.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import xtrack as xt

from joblib import Parallel, delayed
from xfields.ibs._kicks import IBSKick

from pictest._cells import attribute_particle_cells
from pictest._meshgrid import create_mesh, mesh_limits
from pictest._sire import scatter_cell_maxcol_sire, scatter_cell_oneperpart_sire
from pictest._takizuka_abe import (
    scatter_cell_maxcol_takizuka_abe,
    scatter_cell_oneperpart_takizuka_abe,
)

if TYPE_CHECKING:
    from typing import Callable, Literal

_METHOD_TO_SIRE_FUNC: dict[str, Callable] = {
    "maxcol": scatter_cell_maxcol_sire,
    "oneperpart": scatter_cell_oneperpart_sire,
}
_METHOD_TO_TAKIZUKA_ABE_FUNC: dict[str, Callable] = {
    "maxcol": scatter_cell_maxcol_takizuka_abe,
    "oneperpart": scatter_cell_oneperpart_takizuka_abe,
}


# TODO: offer T&A or SIRE as a choice
class IBSParticleInCell(IBSKick):
    r"""
    Beam element that applies IBS effects to particles in a PIC-like
    fashion, simulating the particle-to-particle Coulomb collisions.

    This is an ugly prototype to generate and apply after each element
    in a line, to simulate the IBS that would have happened in that
    previous element.

    Attributes
    ----------
    nx : int
        Number of cells for the meshgrid, in the
        horizontal coordinate.
    ny : int
        Number of cells for the meshgrid, in the
        vertical coordinate.
    nz : int
        Number of cells for the meshgrid, in the
        longitudinal coordinate.
    delta_t : float
        The time step of the IBS effect application,
        in [s]. Obtained from the distance from the
        previous and the particles' velocity (L / v).
    model : str
        Which collision model is used to determine the
        scattering of the particles.
    method : str
        The method used to determine the random draw of
        the collided particle pairs in a cell.
    max_collisions : int | None
        The max number of collisions to perform per cell.
        Will be `None` unless the chosen method needs it.
    scatter_cell : Callable
        The scattering function called for each cell of the
        meshgrid, that will apply Coulomb collisions.
    """

    isthick = False

    def __init__(self, nx: int, ny: int, nz: int, model: Literal["SIRE", "T&A"], cell_method: str, delta_t: float, **kwargs) -> None:
        """
        Initialize the PIC element with necessary properties.

        Parameters
        ----------
        nx : int
            Number of cells for the meshgrid, in the
            horizontal coordinate.
        ny : int
            Number of cells for the meshgrid, in the
            vertical coordinate.
        nz : int
            Number of cells for the meshgrid, in the
            longitudinal coordinate.
        model : str
            Which collision model to use. Valid options are
            as follows (case-insensitive):
                - 'SIRE': an attempt to reproduce the SIRE
                    model by A. Vivoli, which unfortunately
                    was never documented.
                - 'T&A': the Takizuka & Abe binary collision
                    model from their paper of 1977.
        cell_method : str
            The method to use to determine the random draw of
            the collided particle pairs in a cell. Valid options
            are as follows (case-insensitive):
                - 'maxcol': pairs are drawn randomly and collided
                    until a provided maximum number of collisions
                    has been performed. If chosen, one should provide
                    'max_collisions'.
                - 'oneperpart': pairs are drawn randomly once and
                    each particle will collide exactly once. If a
                    cell contains an odd number of particles then
                    one of them will not collide.
        delta_t : float
            The time step of the IBS effect application, in [s].
            To be obtained from the previous element's length and
            the particles' velocity (L / v).
        kwargs : dict
            If the chosen method is 'maxcol', then an additional
            argument 'max_collisions' is expected, as an integer.
            Kwargs are inspected for a boolean 'use_delta' which
            determines if the longitudinal momentum is to be used
            for the meshgrid instead of the zeta coordinate.
        """
        # ----------------------------------------------
        # Make sure we have a valid method, and max_collisions if relevant
        self.model = model.lower()
        self.method = cell_method.lower()
        if self.model not in ("sire", "t&a"):
            raise ValueError("Invalid parameter 'model'. See docstring.")
        if self.method not in ("maxcol", "oneperpart"):
            raise ValueError("Invalid parameter 'method'. See docstring.")
        max_collisions = kwargs.pop("max_collisions", None)
        if self.method == "maxcol" and max_collisions is None:
            raise ValueError("Please provide 'max_collisions' when using 'maxcol'")
        # ----------------------------------------------
        # Store information on using delta or zeta for longitudinal coord
        self._use_delta = kwargs.pop("use_delta", False)
        # ----------------------------------------------
        # Determine the cell scattering function based on the method
        # TODO: based on approach set this to either T&A or SIRE vars
        _method_to_func = {
            "maxcol": scatter_cell_maxcol_sire,
            # "allpairs": scatter_cell_allpairs,
            "oneperpart": scatter_cell_oneperpart_sire,
        }
        cell_scatter_function = _method_to_func[self.method]
        # ----------------------------------------------
        # We store everything
        self.nx: int = nx
        self.ny: int = ny
        self.nz: int = nz
        self.delta_t: float = delta_t
        self.max_collisions: int = max_collisions
        self.scatter_cell: Callable = partial(cell_scatter_function, max_collisions=max_collisions)

    def __repr__(self) -> str:
        collstr = (
            f"\t max_collisions={self.max_collisions} \n" if self.method.lower() == "maxcol" else ""
        )
        return (
            f"{self.__class__.__name__} (method {self.method}) \n" + collstr + f"\t nx={self.nx} \n"
            f"\t ny={self.ny} \n"
            f"\t nz={self.nz} \n"
            f"\t delta_t={self.delta_t} \n"
        )

    def __str__(self) -> str:
        return "IBS Particle in Cell Element"

    def track(self, particles: xt.Particles) -> None:
        """
        Method to split the physical space with a meshgrid,
        attribute particles to cells and for each cell apply
        Coulomb kicks to pairs of particles.

        Parameters
        ----------
        particles : xtrack.Particles
            The particles to apply the IBS kicks to.
        """
        # ----------------------------------------------
        # Determine mesh properties and create mesh
        x_range, y_range, z_range = mesh_limits(particles, self._use_delta)
        meshgrid = create_mesh(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
        )
        # ----------------------------------------------
        # Attribute particles to cells in the mesh
        attributions: np.ndarray = attribute_particle_cells(particles, meshgrid, self._use_delta)
        # ----------------------------------------------
        # Apply kicks according to chosen method, cell per cell (parallelized-ish)
        # Prefer threads as most of our compute is numba and should release the GIL
        _ = Parallel(n_jobs=-2, prefer="threads")(
            delayed(self.scatter_cell)(
                cell,
                attributions,
                meshgrid.toty,
                meshgrid.cell_volume,
                self.delta_t,
                particles,
            )
            for cell in np.unique(attributions)
        )
        # ----------------------------------------------
        # To go sequentially (this is much slower):
        # for cell in np.unique(attributions):
        #     self.scatter_cell(
        #         cell,
        #         attributions,
        #         meshgrid.toty,
        #         meshgrid.cell_volume,
        #         self.delta_t,
        #         particles,
        #     )
