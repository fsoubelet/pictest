"""
Prototyping a compiled version of the PIC.
"""

from pathlib import Path

import xobjects as xo
import xtrack as xt


class MeshGridProperties(xo.Struct):
    nx = xo.Int32
    ny = xo.Int32
    nz = xo.Int32
    minx = xo.Float64
    maxx = xo.Float64
    miny = xo.Float64
    maxy = xo.Float64
    minz = xo.Float64
    maxz = xo.Float64
    delta_x = xo.Float64
    delta_y = xo.Float64
    delta_z = xo.Float64


class PICTRACK(xt.BeamElement):
    r"""
    Beam element that applies IBS effects to particles in a PIC-like
    fashion, simulating the particle-to-particle Coulomb collisions.
    This is the `xsuite`-kernels powered version. Defauls to do the
    'maxcol' scatter routine, no other option.

    This is an ugly prototype to generate and apply after each element
    in a line, to simulate the IBS that would have happened in that
    previous element.

    Xobject Attributes
    ------------------
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
    max_collisions : int
        The max number of collisions to perform per cell.
    """

    iscollective = True
    isthick = False

    _xofields = {
        "nx": xo.Int16,
        "ny": xo.Int16,
        "nz": xo.Int16,
        "delta_t": xo.Float64,
        "max_collisions": xo.Int16,
        "_attributions": xo.Int16[:],
    }

    _extra_c_sources = [
        Path(__file__).parent / "kernels" / "_pic.h",
        Path(__file__).parent / "kernels" / "_attribution.h",
    ]

    # Some extra kernels we define - per particle (for cell attributions)
    _per_particle_kernels = {
        "attribute_cells": xo.Kernel(
            c_name="PICTRACK_attribute_cells",
            args=[
                xo.Arg(xo.ThisClass, name="el"),
                xo.Arg(MeshGridProperties, name="mesh"),
            ],
        )
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
