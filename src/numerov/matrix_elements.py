import logging

import numpy as np
import scipy.integrate

from numerov.rydberg import RydbergState

logger = logging.getLogger(__name__)


def radial_matrix_element(
    state1: RydbergState,
    state2: RydbergState,
    r_power: int = 0,
) -> float:
    r"""Calculate the radial matrix element between two Rydberg states.

    Computes the integral

    .. math::
        \int_{0}^{\infty} dr r^2 r^\kappa R_1(r) R_2(r)
        = a_0^\kappa \int_{0}^{\infty} dx x^\kappa \tilde{u}_1(x) \tilde{u}_2(x)
        = a_0^\kappa \int_{0}^{\infty} dz 2 z^{2 + 2\kappa} w_1(z) w_2(z)

    where R_1 and R_2 are the radial wavefunctions of the two states
    and w(z) = z^{-1/2} \tilde{u}(z^2) = (r/_a_0)^{1/4} \sqrt{a_0} r R(r).

    Args:
        state1: First Rydberg state
        state2: Second Rydberg state
        r_power: Power of r in the matrix element
        (default=0, this corresponds to the overlap integral \int dr r^2 R_1(r) R_2(r))

    Returns:
        float: The radial matrix element

    """
    # Ensure both states have been integrated
    if not hasattr(state1, "w_list") or not hasattr(state2, "w_list"):
        raise ValueError("Both states must be integrated before calculating matrix elements")

    # Get the z coordinates
    z1 = state1.z_list
    z2 = state2.z_list

    # Check if grid step sizes are compatible
    dz1 = z1[1] - z1[0]
    dz2 = z2[1] - z2[0]
    if not np.isclose(dz1, dz2):
        raise ValueError("Both states must be integrated with the same step size")

    # Find overlapping grid range
    zmin = max(z1[0], z2[0])
    zmax = min(z1[-1], z2[-1])
    if zmax <= zmin:
        logger.debug("No overlapping grid points between states, returning 0")
        return 0

    # Select overlapping points
    tol = 1e-6
    mask1 = (z1 >= zmin - tol) & (z1 <= zmax + tol)
    mask2 = (z2 >= zmin - tol) & (z2 <= zmax + tol)
    z1 = z1[mask1]
    z2 = z2[mask2]

    if len(z1) != len(z2) or not np.allclose(z1, z2):
        raise ValueError(f"Overlapping grid points are not equal: {z1=} != {z2=}")

    wf1 = state1.w_list[mask1]
    wf2 = state2.w_list[mask2]

    if not np.isclose(dz1, dz2):
        raise ValueError("Both states must be integrated with the same step size")

    # integrand in terms of z and w(z), see docstring
    integrand = 2 * np.power(z1, 2 * r_power + 2) * wf1 * wf2

    # Integrate using Simpson's rule
    return float(scipy.integrate.simpson(integrand, z1))
