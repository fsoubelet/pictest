"""
Installation functionality.
"""

from __future__ import annotations

import numpy as np
import xtrack as xt

from scipy.constants import c
from xtrack.progress_indicator import progress

from pictest._pic import IBSParticleInCell


def install_ibs_pic(
    line: xt.Line | None = None,
    *,
    _buffer=None,
    particle_ref: xt.Particles | None = None,
    num_ibs_interactions: int = 100,
    s_pics: np.ndarray | None = None,
    **kwargs,
) -> tuple[list[IBSParticleInCell], list[float]]:
    """
    Install IBS ParticleInCell elements in an `xtrack.Line`.
    This steals the exact logic of the installation of the
    Space Charge interactions.

    Parameters
    ----------
    line : xtrack.Line
        Line in which the IBSParticleInCell elements
        are installed.
    particle_ref : xpart.Particles (optional)
        Reference particle for the spacecharge elements.
        The line's particle ref will be used if this is
        not provided.
    num_ibs_interactions : int
        The number of PIC elements to insert in the line.
        They are by default equi-distantly spaced through
        the line when intalled, unless specific positions
        are provided (see below).
    s_pics : ArrayLike (optional)
        Position of the IBS PIC elements. If not given,
        they will be equally spaced through the line.
    **kwargs : dict
        All keyword arguments are passed on to the
        `IBSParticleInCell` element creation. If not
        provided, then the `delta_t` values will be
        determined automatically from the positions
        of the elements in the line. Expected are `nx`,
        `ny`, `nz`,  `method` and potentially also
        `max_collisions`. See details in the docstring
        of the `IBSParticleInCell` class.

    Returns
    -------
    ibs_elements : list
        A list of the installed elements.
    positions : list
        A list of the exact S positions elements were inserted at.
    """
    # ----------------------------------------------
    # Take care of the kwargs we expect, with defaults
    nx = kwargs.pop("nx", 10)
    ny = kwargs.pop("ny", 10)
    nz = kwargs.pop("nz", 10)
    model = kwargs.pop("model")  # should be provided
    method = kwargs.pop("method")  # should be provided
    kdeltat = kwargs.pop("delta_t", None)  # to override
    # if method.lower() == "maxcol":
    max_collisions = kwargs.pop("max_collisions")
    # ----------------------------------------------
    # Handle the possibly existing line tracker
    if _buffer is None:
        if not line._has_valid_tracker():
            line.build_tracker(compile=False)  # put everything in the same buffer
        _buffer = line._buffer
    line.discard_tracker()  # we will reinstore later
    # ----------------------------------------------
    # THE FOLLOWING IS A COPY-PASTED ADAPTED FROM THE
    # SPACE CHARGE INSTALLATION FUNCTION IN XFIELDS
    # ----------------------------------------------
    # Handle the particle_ref
    if particle_ref is None:
        if line.particle_ref is None:
            raise AttributeError("A reference particle is needed")
        particle_ref = line.particle_ref
    speed = c * np.sqrt(1 - 1 / particle_ref.gamma0[0] ** 2)  # [m/s]
    # ----------------------------------------------
    # Generate IBS PIC positions
    if s_pics is None:
        s_pics = np.linspace(0, line.get_length(), num_ibs_interactions + 1)[:-1]
    # ----------------------------------------------
    # Create IBS elements to be inserted in the line
    ibs_elements = []
    ibs_names = []
    insertions = []
    for ii in progress(range(len(s_pics)), desc="Installing IBS PIC elements"):
        s_pos = s_pics[ii]
        if ii == 0:  # first element installed, we want distance from last pic in lattice
            dist_from_last_ibs = s_pos + (line.get_length() - s_pics[ii - 1])
        else:  # just distance from previous one
            dist_from_last_ibs = s_pos - s_pics[ii - 1]
        deltat = dist_from_last_ibs / speed  # [s]
        # We create the element and set important properties
        pic_name = f"ibs_pic_{ii:d}"
        pic_element = IBSParticleInCell(
            nx=nx,
            ny=ny,
            nz=nz,
            model=model,
            cell_method=method,
            delta_t=kdeltat if kdeltat is not None else deltat,
            max_collisions=max_collisions,
        )
        pic_element._name = pic_name
        pic_element._scale_strength = 1
        # Store these for later to be inserted
        ibs_elements.append(pic_element)
        ibs_names.append(pic_name)
        insertions.append((s_pos, [(ibs_names[-1], ibs_elements[-1])]))
    # ----------------------------------------------
    # Insert the created elements in the line
    line._insert_thin_elements_at_s(insertions)
    # ----------------------------------------------
    # Now we run a twiss (4D) which includes the inserted PICs
    # and we provide it to all inserted elements
    twiss = line.twiss(method="4d")
    for pic in ibs_elements:
        pic._twiss = twiss
    # ----------------------------------------------
    # Restore the buffer from before and the elements
    line.build_tracker(_buffer=_buffer)
    return ibs_elements, [insert[0] for insert in insertions]  # the used s positions
