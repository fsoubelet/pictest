"""
Visualisation functions for the package.
"""

from __future__ import annotations

import numpy as np
import xtrack as xt

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Path3DCollection

from pictest._meshgrid import MeshGrid

Range = tuple[float, float]

# ------ General Utilities ----- #


def maybe_get_ax(**kwargs):
    """
    .. versionadded:: 1.0.0

    Convenience function to get the axis, regardless of whether or
    not it is provided to the plotting function itself. It used to
    be that the first argument of plotting functions in this package
    had to be the 'axis' object, but that's no longer the case.

    Parameters
    ----------
    *args
        The arguments passed to the plotting function.
    **kwargs
        The keyword arguments passed to the plotting function.

    Returns
    -------
    tuple[matplotlib.axes.Axes, tuple, dict]
        The `~matplotlib.axes.Axes` object to plot on, as well as the args
        and kwargs (without the 'ax' argument if it initially was present).
        If no axis was provided, then it will be created with a call to
        `matplotlib.pyplot.gca`.

    Example
    -------
        This is to be called at the beginning of your plotting functions:

        .. code-block:: python

            def my_plotting_function(*args, **kwargs):
                ax, kwargs = maybe_get_ax(**kwargs)
                # do stuff with ax
                ax.plot(*args, **kwargs)
    """
    if "ax" in kwargs:
        ax = kwargs.pop("ax")
    elif "axis" in kwargs:
        ax = kwargs.pop("axis")
    else:
        ax = plt.gca()
    return ax, dict(kwargs)


# ------ Particles Plotting Functions ----- #


def plot_particles_3dtrace(
    particles: xt.Particles, plot_delta: bool = False, **kwargs
) -> Path3DCollection:
    """
    Plot the particles on the given or current axis, which
    should be a 3D projection.

    Parameters
    ----------
    particles : xtrack.Particles
        The particles distribution object to plot.
    kwargs : dict
        Keyword arguments are used to look for the axis
        to act on. If no axis is given, the current axis
        is used. Remaining keyword arguments are passed
        to the scatter function.

    Returns
    -------
    p : Path3DCollection
        The 3D path collection object returned by the
        axis' scatter method.
    """
    ax, kwargs = maybe_get_ax(**kwargs)
    if plot_delta is False:
        p = ax.scatter(particles.x, particles.y, particles.zeta, **kwargs)
        ax.set_zlabel(r"zeta [-]")
    else:
        p = ax.scatter(particles.x, particles.y, particles.delta, **kwargs)
        ax.set_zlabel(r"$\delta$ [-]")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    return p


def zoomout_axes_limits(particles: xt.Particles = None, nsigmas: float = None, **kwargs) -> None:
    """
    Set limits as directly given or as sigmas.

    Parameters
    ----------
    particles : xtrack.Particles
        The particles distribution object to plot.
    kwargs : dict
        Keyword arguments are used to look for the axis
        to act on. If no axis is given, the current axis
        is used.
    """
    ax, kwargs = maybe_get_ax(**kwargs)
    xlim = kwargs.get("xlim", None)
    ylim = kwargs.get("ylim", None)
    zlim = kwargs.get("zlim", None)
    xlim = xlim or _nsigma_ranges(particles, nsigmas)[0]
    ylim = ylim or _nsigma_ranges(particles, nsigmas)[1]
    zlim = zlim or _nsigma_ranges(particles, nsigmas)[2]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)


# ------ MeshGrid Plotting Functions ----- #


def plot_meshgrid_points(meshgrid: MeshGrid, **kwargs) -> None:
    """
    Plot the meshgrid points on the given or current
    axis, which should be a 3D projection.

    Parameters
    ----------
    meshgrid : MeshGrid
        The meshgrid object to plot.
    kwargs : dict
        Keyword arguments are used to look for the axis
        to act on. If no axis is given, the current axis
        is used. Remaining keyword arguments are passed
        to the scatter function.
    """
    ax, kwargs = maybe_get_ax(**kwargs)
    ax.scatter(meshgrid.XX, meshgrid.YY, meshgrid.ZZ, **kwargs)


def plot_meshgrid_voxels(meshgrid: MeshGrid, **kwargs) -> None:
    """
    Plot the meshgrid cells as voxels on the given or
    current axis, which should be a 3D projection.

    Parameters
    ----------
    meshgrid : MeshGrid
        The meshgrid object to plot.
    kwargs : dict
        Keyword arguments are used to look for the axis
        to act on. If no axis is given, the current axis
        is used. Remaining keyword arguments are passed
        to the voxels function.
    """
    ax, kwargs = maybe_get_ax(**kwargs)
    filled = kwargs.pop("filled", None)
    # We are using coordinates for the voxels so we need to provide 'filled'
    # which is a mesh of True values with a shape 1 less than the meshgrid
    filled = np.ones(tuple(np.asarray(meshgrid.XX.shape) - 1), dtype=bool)
    ax.voxels(meshgrid.XX, meshgrid.YY, meshgrid.ZZ, filled=filled, **kwargs)


# ------ Helpers ----- #


def _nsigma_ranges(particles: xt.Particles, nsigmas: float) -> tuple[Range, Range, Range]:
    """
    Return the x, y and z ranges corresponding to the 'sigmas'
    for the particle distrubition. These are the (minval, maxval)
    pairs for each dimension.

    Parameters
    ----------
    particles : xtrack.Particles
        The particles distribution object to determine from.
    nsigmas : float
        The number of sigmas to find the limits for.

    Returns
    -------
    ranges : tuple[Range, Range, Range]
        The x, y and z ranges as (minval, maxval) pairs.
    """
    # Determining the range by using n sigma deviation from the mean of a coordinate
    xmin = particles.x.mean() - nsigmas * particles.x.std()
    xmax = particles.x.mean() + nsigmas * particles.x.std()
    ymin = particles.y.mean() - nsigmas * particles.y.std()
    ymax = particles.y.mean() + nsigmas * particles.y.std()
    zmin = particles.zeta.mean() - nsigmas * particles.zeta.std()
    zmax = particles.zeta.mean() + nsigmas * particles.zeta.std()
    return (xmin, xmax), (ymin, ymax), (zmin, zmax)
