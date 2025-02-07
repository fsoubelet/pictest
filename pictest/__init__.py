import contextlib
import sys

from ._meshgrid import MeshGrid, create_mesh, mesh_limits
from ._pic import IBSParticleInCell
from ._plotting import plot_meshgrid_points, plot_meshgrid_voxels, plot_particles_3dtrace
from ._setup import install_ibs_pic

__version__ = "0.1.0"


class _DummyFile(object):
    def write(self, x):
        pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = _DummyFile()
    yield
    sys.stdout = save_stdout


__all__ = [
    "MeshGrid",
    "create_mesh",
    "IBSParticleInCell",
    "install_ibs_pic",
    "mesh_limits",
    "nostdout",
    "plot_meshgrid_points",
    "plot_meshgrid_voxels",
    "plot_particles_3dtrace",
]
