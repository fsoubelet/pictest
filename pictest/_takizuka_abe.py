"""
Takizuka and Abe related functions: numba JIT-compiled code to determine
various parameters that appear in the collisions computations.
"""

from __future__ import annotations

import itertools
import random

from typing import TYPE_CHECKING

import numba
import numpy as np

from scipy.constants import c, epsilon_0

from pictest._cells import find_index_of_all_particles_in_given_cell

if TYPE_CHECKING:
    import xtrack as xt


# ----- Cell Scattering Functions ----- #


def scatter_cell_maxcol_takizuka_abe(
    cell_number: int,
    attributions: np.ndarray,
    volume: float,
    coulog: float,
    delta_t: float,
    particles: xt.Particles,
    max_collisions: int,
) -> None:
    """
    Considers a single cell of the meshgrid and applies collisions
    to randomly chosen pairs of particles in the cell. Stops when
    the maximum number of collisions has been reached.

    This draws random pairs with replacement for each collision,
    until we reach the max number of collisions to do in the cell.

    Parameters
    ----------
    cell_number : int
        The identifier of the cell to consider.
    attributions : np.ndarray
        The array of cell attributions for each particle, as given
        by the `attribute_particle_cells_like_sire` function.
    volume : float
        The volume of the cell to consider, in [m^3]. All cells
        have the same volume as the meshgrid is uniform.
    coulog : float
        The Coulomb logarithm for the whole bunch.
    delta_t : float
        The time step of the IBS effect application, in [s].
        Should depend on the element length and the particle
        velocity.
    max_collisions : int
        The maximum number of collisions to apply in the cell.
    particles : xtrack.Particles
        The particles distribution object to consider and act on.
    """
    # ----------------------------------------------
    # Get some cell-specific parameters we will need
    cell_particles = find_index_of_all_particles_in_given_cell(cell_number, attributions)
    n_macroparts: int = cell_particles.size  # number of parts in this cell
    cell_particles = list(cell_particles)  # INDICES - need as list for sampling
    print(f"CELL {cell_number:d}, has {n_macroparts:d} parts")
    # ----------------------------------------------
    # Determine the number of collisions to do in this cell. If there are more
    # max collisions than particles, we do 1 less than particles (if there are 2
    # parts, there will be 1 collision), otherwise we do the user-provided max
    n_collisions = n_macroparts - 1 if max_collisions >= n_macroparts else max_collisions
    if n_collisions == 0:  # don't waste time and risk 0-division error
        return
    # ----------------------------------------------
    # We determine the n_l parameter for the collision function
    # which is define in the "Determination of pairs" paragraph
    weight = particles.weight[0]  # same for all, accounts for real part / macropart
    Ni = n_macroparts * weight  # only one species, Ni = Ne
    N0 = np.sum(particles.state > 0) * weight  # all alive parts, "particle number in a cloud"
    V0 = volume  # happens to be the same for all cells
    n_l = Ni * N0 / V0  # since Ni = Ne
    # ----------------------------------------------
    # Now we collide as long as we have to
    while n_collisions > 0:
        part1, part2 = random.sample(cell_particles, k=2)  # choose a pair
        collide_particle_pair_takizuka_abe(
            idx1=part1,
            idx2=part2,
            coulog=coulog,
            delta_t=delta_t,
            n_l=n_l,
            particles=particles,
        )
        # That's one less collision to do
        n_collisions -= 1


def scatter_cell_oneperpart_takizuka_abe(
    cell_number: int,
    attributions: np.ndarray,
    volume: float,
    coulog: float,
    delta_t: float,
    particles: xt.Particles,
    **kwargs,
) -> None:
    """
    Considers a single cell of the meshgrid and applies collisions
    to randomly chosen pairs of particles in the cell.

    This draws random pairs without replacement for each collision,
    and does exactly one collision per particle.

    Parameters
    ----------
    cell_number : int
        The identifier of the cell to consider.
    attributions : np.ndarray
        The array of cell attributions for each particle, as given
        by the `attribute_particle_cells_like_sire` function.
    volume : float
        The volume of the cell to consider, in [m^3]. All cells
        have the same volume as the meshgrid is uniform.
    coulog : float
        The Coulomb logarithm for the whole bunch.
    delta_t : float
        The time step of the IBS effect application, in [s].
        Should depend on the element length and the particle
        velocity.
    particles : xtrack.Particles
        The particles distribution object to consider and act on.
    kwargs : dict
        Not used, this is merely so the function does
        not crash if given 'max_collisions'.
    """
    # ----------------------------------------------
    # Get some cell-specific parameters we will need
    cell_particles = find_index_of_all_particles_in_given_cell(cell_number, attributions)
    n_macroparts: int = cell_particles.size  # number of parts in this cell
    cell_particles = list(cell_particles)  # INDICES - need as list for sampling
    random.shuffle(cell_particles)  # for randomness
    # ----------------------------------------------
    # Determine an ensemble of pairs, for 1 collision per particle
    # We just batch 2 by 2 since we shuffled (for randomness) above
    collided_pairs = list(itertools.batched(cell_particles, 2))
    n_collisions = len(collided_pairs)
    if n_collisions == 0:  # don't waste time and risk 0-division error
        return
    # ----------------------------------------------
    # We determine the n_l parameter for the collision function
    # which is define in the "Determination of pairs" paragraph
    weight = particles.weight[0]  # same for all, accounts for real part / macropart
    Ni = n_macroparts * weight  # only one species, Ni = Ne
    N0 = np.sum(particles.state > 0) * weight  # all alive parts, "particle number in a cloud"
    V0 = volume  # happens to be the same for all cells
    n_l = Ni * N0 / V0  # since Ni = Ne
    # ----------------------------------------------
    # Now we collide for every single possible pair
    try:
        for part1, part2 in collided_pairs:
            collide_particle_pair_takizuka_abe(
                idx1=part1,
                idx2=part2,
                coulog=coulog,
                delta_t=delta_t,
                n_l=n_l,
                particles=particles,
            )
    except ValueError:  # happens for an odd number of particles (last pair has no 'part2')
        pass


# ----- Particle Pair Collision Function ----- #


def collide_particle_pair_takizuka_abe(
    idx1: int, idx2: int, coulog: float, n_l: float, delta_t: float, particles: xt.Particles
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
    coulog : float64
        The Coulomb logarithm for the whole bunch.
    n_l : float64
        The lower density between n_alpha and n_beta.
        These are defined in the "determination of pairs"
        paragraph. Note that n_alpha = n_beta = n_l (only
        have one species of particles).
    delta_t : float
        The time step of the IBS interaction, in [s].
    particles : xt.Particles
        The `xtrack.Particles` object with the particles
        information, to be directly modified.
    """
    print(f"Colliding particles at idx {idx1} and {idx2}")
    # ----------------------------------------------
    # Get some global properties - nly have one particle
    # species so we take properties of the first particle
    mass0 = particles.mass0
    q0 = particles.q0
    # ----------------------------------------------
    # Get the particle properties needed for the collision
    px1, px2 = particles.px[idx1], particles.px[idx2]
    py1, py2 = particles.py[idx1], particles.py[idx2]
    delta1, delta2 = particles.delta[idx1], particles.delta[idx2]
    # ----------------------------------------------
    # Compute the momentum deltas (compiled code)
    delta_px, delta_py, delta_pz = takizuka_abe_collision_deltas(
        px1, px2, py1, py2, delta1, delta2, q0, mass0, coulog, delta_t, n_l
    )
    # ----------------------------------------------
    # Apply the deltas to the particles (add to part1, remove from part2)
    particles.px[idx1] += delta_px
    particles.py[idx1] += delta_py
    particles.delta[idx1] += delta_pz
    particles.px[idx2] -= delta_px
    particles.py[idx2] -= delta_py
    particles.delta[idx2] -= delta_pz


# ----- Collision Deltas Function ----- #


@numba.jit
def takizuka_abe_collision_deltas(
    px1: numba.float64,  # type: ignore
    px2: numba.float64,  # type: ignore
    py1: numba.float64,  # type: ignore
    py2: numba.float64,  # type: ignore
    delta1: numba.float64,  # type: ignore
    delta2: numba.float64,  # type: ignore
    q0: numba.float64,  # type: ignore
    mass0: numba.float64,  # type: ignore
    coulog: numba.float64,  # type: ignore
    delta_t: numba.float64,  # type: ignore
    n_l: numba.float64,  # type: ignore
) -> tuple[numba.float64, numba.float64, numba.float64]:  # type: ignore
    """
    Compute momentum deltas for a two-article Coulomb collision, trying to stay
    exactly on the formulae from Takizuka and Abe. This is a compiled function
    to be used in the main loop of the PIC simulation.

    Notes
    -----
    In here I will be using the exact variable names of the T&A paper,
    but try to comment as much as possible to explain things.

    Parameters
    ----------
    px1 : float64
        Horizontal momentum of the first particle of the pair.
    px2 : float64
        Horizontal momentum of the second particle of the pair.
    py1 : float64
        Vertical momentum of the first particle of the pair.
    py2 : float64
        Vertical momentum of the second particle of the pair.
    delta1 : float64
        Momentum deviation of the first particle of the pair.
    delta2 : float64
        Momentum deviation of the second particle of the pair.
    q0 : float64
        The charge of the particles in the pair (same species),
        in [e].
    mass0 : float64
        The mass of the particles in the pair (same species), in
        [eV] as `xtrack` gives us.
    coulog : float64
        The Coulomb logarithm for the whole bunch.
    delta_t : float64
        The time interval for the IBS interaction, in [s].
    n_l : float64
        The lower density between n_alpha and n_beta. These
        are defined in the "determination of pairs" paragraph.
        Note that n_alpha = n_beta = n_l (only one species).

    Returns
    -------
    deltas : tuple[float64, float64, float64]
        The momentum deltas in horizontal, vertical
        and longitudinal momenta. These are already the
        amount to apply to particles.
    """
    # ----------------------------------------------
    # First define the 'u' variables from Eq. (1)
    # We divide by mass because xtrack uses momentum
    _ev_to_J = 1.602176634 * 10**-19  # conversion factor from eV to J
    _eV_to_kg = _ev_to_J / c**2  # conversion factor from eV to kg
    mass_kg = mass0 * _eV_to_kg  # we want mass in [g]
    # mass_kg = mass0  # TODO: revert this test & uncomment above
    ux = (px1 - px2) / mass_kg  # ux = vx1 - vx2
    uy = (py1 - py2) / mass_kg  # uy = vy1 - vy2
    uz = (delta1 - delta2) / mass_kg  # uz = vz1 - vz2
    # ----------------------------------------------
    # We compute m_alpha_beta from Eq (6). We only have
    # one species so m_alpha = m_beta = mass_g
    m_alpha_beta = mass_kg / 2  # simplified mass_g**2 / (2 * mass_g)
    # ----------------------------------------------
    # Now we compute phi, theta and then u from Eq. (2)
    # See relevant function docstrings for information
    phi = _compute_phi(ux, uy)
    theta = _compute_theta(ux, uy, uz, phi)
    u = np.sin(theta) * np.cos(phi) * ux + np.sin(theta) * np.sin(phi) * uy + np.cos(theta) * uz
    # ----------------------------------------------
    # We draw the polar collision angle PHI from Eq (3)
    # (from a uniform distribution between 0 and 2pi)
    PHI: float = _draw_PHI()
    # ----------------------------------------------
    # We draw a value for delta according to Eq (8a)
    # and then plug its value into Eq (7a) for THETA
    coulog = 20  # TODO: remove after testing
    delta: float = _draw_delta(q0, m_alpha_beta, coulog, delta_t, n_l, u)
    THETA: float = np.arcsin(2 * delta / (1 + delta**2))
    # delta = None  # TODO: revert this test & uncomment above
    # THETA: float = _draw_THETA()  # TODO: revert this test & uncomment above
    # ----------------------------------------------
    # We compute U_T defined below Eq (4d)
    u_t: float = np.sqrt(ux**2 + uy**2)
    # ----------------------------------------------
    # Now, if u_t != 0 we compute delta_ux, delta_uy
    # and delta_uz from Eq (4.b), Eq (4.c) & Eq (4.d)
    if u_t != 0:
        delta_ux = (
            ux * uz * np.sin(THETA) * np.cos(PHI) / u_t
            - uy * u * np.sin(THETA) * np.sin(PHI) / u_t
            - ux * (1 - np.cos(THETA))
        )
        delta_uy = (
            uy * uz * np.sin(THETA) * np.cos(PHI) / u_t
            + ux * u * np.sin(THETA) * np.sin(PHI) / u_t
            - uy * (1 - np.cos(THETA))
        )
        delta_uz = -1 * u_t * np.sin(THETA) * np.cos(PHI) - uz * (1 - np.cos(THETA))
    # ----------------------------------------------
    # Otherwise we compute to their alternative forms
    # as in Eq (4b'), Eq (4c') and Eq (4d')
    else:
        delta_ux = u * np.sin(THETA) * np.cos(PHI)
        delta_uy = u * np.sin(THETA) * np.sin(PHI)
        delta_uz = u * (np.cos(THETA) - 1)
    # ----------------------------------------------
    # These are deltas to apply to velocities (see Eq (5a))
    # so we convert to deltas to apply to momenta (used in
    # Xsuite). See notebook to figure out the form.
    res_delta_px = m_alpha_beta * delta_ux
    res_delta_py = m_alpha_beta * delta_uy
    res_delta_pz = m_alpha_beta * delta_uz
    # ----------------------------------------------
    # And finally we can return the computed momentum deltas
    print("phi=", phi, "theta=", theta, "u=", u, "PHI=", PHI, "coulog=", coulog, "delta=", delta, "THETA=", THETA)
    print("delta_px=", res_delta_px, "delta_py=", res_delta_py, "delta_pz=", res_delta_pz)
    return res_delta_px, res_delta_py, res_delta_pz


# ----- Private Helpers for Takizuka and Abe ----- #


@numba.jit
def _compute_phi(
    ux: numba.float64,  # type: ignore
    uy: numba.float64,  # type: ignore
) -> numba.float64:  # type: ignore
    """
    Solves the second system of Eq (2) in Takizuka and Abe to
    determine the value of phi. An important consideration is
    that phi - as an angle - has to be positive. We will then
    try the first root of the equation and if its value is not
    positive we will return the second root instead.

    See my solving notebook for the derivation of the formula.
    The roots' analytical forms were determined with sympy and
    the validity of the written solutions (decomposed assembled
    terms) was verified just the same.

    Parameters
    ----------
    ux : float64
        The horizontal velocity difference of the particles.
    uy : float64
        The vertical velocity difference of the particles.

    Returns
    -------
    phi : float64
        The value of phi, the angle between ux and ut, in radians.
    """
    # Both roots differ by only the sign in front
    # of the square root term in the atan
    # ----------------------------------------------
    # Define term used several times to avoid recomputing
    sqrt_term = np.sqrt(ux**2 + uy**2)
    # ----------------------------------------------
    # Attempt the first root of the equation
    first_root = -2 * np.arctan((ux - sqrt_term) / uy)
    if first_root >= 0:
        return first_root
    # ----------------------------------------------
    # Otherwise attempt the second root of the equation
    second_root = -2 * np.arctan((ux + sqrt_term) / uy)
    return second_root


@numba.jit
def _compute_theta(
    ux: numba.float64,  # type: ignore
    uy: numba.float64,  # type: ignore
    uz: numba.float64,  # type: ignore
    phi: numba.float64,  # type: ignore
) -> numba.float64:  # type: ignore
    """
    Solves the first system of Eq (2) in Takizuka and Abe to
    determine the value of theta. An important consideration
    is that theta - as an angle - has to be positive. We will
    then try the first root of the equation and if it is not
    positive we will return the second root instead.

    It is assumed that phi has already been solved for with
    the second system of Eq (2) in Takizuka and Abe, since
    it is an input to this function.

    See my solving notebook for the derivation of the formula.
    The roots' analytical forms were determined with sympy and
    the validity of the written solutions (decomposed assembled
    terms) was verified just the same.

    Parameters
    ----------
    ux : float64
        The horizontal velocity difference of the particles.
    uy : float64
        The vertical velocity difference of the particles.
    uz : float64
        The longitudinal velocity difference of the particles.
    phi : float64
        The angle between ux and ut, see figure
    """
    # Both roots differ by only the sign in front
    # of the square root term in the atan
    # ----------------------------------------------
    # Define term used everywhere to avoid recomputing
    tan_phi_2 = np.tan(phi / 2)  # compute only once
    # ----------------------------------------------
    # Define terms one by one first for clarity
    term_one = uz * tan_phi_2**2
    term_two = ux**2 * tan_phi_2**4
    term_three = 2 * ux**2 * tan_phi_2**2
    term_four = 4 * ux * uy * tan_phi_2**3
    term_five = 4 * ux * uy * tan_phi_2
    term_six = 4 * uy**2 * tan_phi_2**2
    term_seven = uz**2 * tan_phi_2**4
    term_eight = 2 * uz**2 * tan_phi_2**2
    # ----------------------------------------------
    # Define some common terms to both roots
    sqrt_term = np.sqrt(
        term_two
        - term_three
        + ux**2
        - term_four
        + term_five
        + term_six
        + term_seven
        + term_eight
        + uz**2
    )
    denominator = -ux * tan_phi_2**2 + ux + 2 * uy * tan_phi_2
    # ----------------------------------------------
    # Attempt the first root of the equation
    first_root = -2 * np.atan((term_one + uz - sqrt_term) / denominator)
    if first_root >= 0:
        return first_root
    # ----------------------------------------------
    # Otherwise attempt the second root of the equation
    second_root = -2 * np.atan((term_one + uz + sqrt_term) / denominator)
    return second_root


@numba.jit
def _draw_PHI() -> numba.float64:  # type: ignore
    """
    Draws a random value for the azimuthal angle PHI,
    first seen in Eq (3) in Takizuka and Abe's paper
    and described in the sentence just below.

    Returns
    -------
    PHI : float64
        A random number between 0 and 2pi.
    """
    return np.random.uniform(0, 2 * np.pi)


@numba.jit
def _draw_delta(
    q0: numba.float64,  # type: ignore
    m_alpha_beta: numba.float64,  # type: ignore
    coulog: numba.float64,  # type: ignore
    delta_t: numba.float64,  # type: ignore
    n_l: numba.float64,  # type: ignore
    u: numba.float64,  # type: ignore
) -> numba.float64:  # type: ignore
    """
    Draws a random value for the variable delta, which is
    used to later determine the scattering angle THETA. Its
    properties are described in Eq (8a) in Takizuka and Abe's
    paper.

    Parameters
    ----------
    q0 : float64
        The charge of the particles in the pair (same species),
        in [e].
    m_alpha_beta : float64
        The m_alpha_beta term of Eq (6) for the pair of
        collided particles. In our case of same-species
        this resolves to mass_g / 2. Units [-].
    coulog : float64
        The Coulomb logarithm for the whole bunch.
    delta_t : float64
        The time interval for the IBS interaction, in [s].
    n_l : float64
        The lower density between n_alpha and n_beta. These
        are defined in the "determination of pairs" paragraph.
        Note that n_alpha = n_beta = n_l (only one species).
    u : float64
        The transverse velocity of the particles.

    Returns
    -------
    delta : float64
        A random number from the relevant distribution.
    """
    # ----------------------------------------------
    # We compute the variance as described by Eq. (8a)
    # Remember e_alpha = e_beta = q0 (only one species)
    variance = (delta_t * (q0**4 * n_l * coulog)) / (
        8 * np.pi * epsilon_0**2 * m_alpha_beta**2 * u**3
    )
    # ----------------------------------------------
    # From the variance we get stdev and draw delta
    scale = np.sqrt(variance)  # standard deviation
    return np.random.normal(0, scale=scale)


@numba.jit
def _draw_THETA() -> numba.float64:  # type: ignore
    """
    Traditionnally we would determine the properties
    of the Gaussian distribution from which to draw
    delta, and then compute theta from the drawn
    variable as an arcsin. Considering arcsing yields
    values between -pi/2 and pi/2, for or a simplified
    (and faster) model we can draw THETA via a uniform
    draw from this range.

    Returns
    -------
    THETA : float64
        A random number between -pi/2 and pi/2.
    """
    return np.random.uniform(-np.pi / 2, np.pi / 2)
