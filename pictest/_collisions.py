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
