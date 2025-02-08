import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.integrate

if TYPE_CHECKING:
    from numerov.rydberg import RydbergState

logger = logging.getLogger(__name__)


def calc_radial_matrix_element(
    state1: "RydbergState",
    state2: "RydbergState",
    r_power: int = 0,
    integration_method: Literal["simpson", "trapezoid"] = "trapezoid",
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
        integration_method: Integration method to use, one of "simpson" or "trapezoid" (default="simpson")

    Returns:
        float: The radial matrix element in atomic units.

    """
    # Make sure the wavefunctions are integrated before accessing the grid
    wf1 = state1.wavefunction
    wf2 = state2.wavefunction
    return calc_radial_matrix_element_from_w_z(
        wf1.grid.zlist, wf1.wlist, wf2.grid.zlist, wf2.wlist, r_power, integration_method
    )


def calc_radial_matrix_element_from_w_z(
    z1: np.ndarray,
    w1: np.ndarray,
    z2: np.ndarray,
    w2: np.ndarray,
    r_power: int = 0,
    integration_method: Literal["simpson", "trapezoid"] = "simpson",
) -> float:
    r"""Calculate the radial matrix element of two wavefunctions w1(z1) and w2(z2).

    Computes the integral

    .. math::
        \int_{0}^{\infty} dz 2 z^{2 + 2\kappa} w_1(z) w_2(z)
        = \int_{0}^{\infty} dx x^\kappa \tilde{u}_1(x) \tilde{u}_2(x)
        = a_0^{-\kappa} \int_{0}^{\infty} dr r^2 r^\kappa R_1(r) R_2(r)

    where R_1 and R_2 are the radial wavefunctions of the two states
    and w(z) = z^{-1/2} \tilde{u}(z^2) = (r/_a_0)^{1/4} \sqrt{a_0} r R(r).

    Args:
        z1: z coordinates of the first wavefunction
        w1: w(z) values of the first wavefunction
        z2: z coordinates of the second wavefunction
        w2: w(z) values of the second wavefunction
        r_power: Power of r in the matrix element
        (default=0, this corresponds to the overlap integral \int dr r^2 R_1(r) R_2(r))
        integration_method: Integration method to use, one of "simpson" or "trapezoid" (default="simpson")

    Returns:
        float: The radial matrix element

    """
    # Find overlapping grid range
    zmin = max(z1[0], z2[0])
    zmax = min(z1[-1], z2[-1])
    if zmax <= zmin:
        logger.debug("No overlapping grid points between states, returning 0")
        return 0

    # Select overlapping points
    tol = (z1[1] - z1[0]) / 2
    mask1 = (z1 >= zmin - tol) & (z1 <= zmax + tol)
    mask2 = (z2 >= zmin - tol) & (z2 <= zmax + tol)
    z1 = z1[mask1]
    z2 = z2[mask2]
    wf1 = w1[mask1]
    wf2 = w2[mask2]

    tol = 1e-10
    assert len(z1) == len(z2), f"Length mismatch: {len(z1)=} != {len(z2)=}"
    assert z1[0] - z2[0] < tol, f"First point mismatch: {z1[0]=} != {z2[0]=}"
    assert z1[1] - z2[1] < tol, f"Step size mismatch: {z1[0]=} != {z2[0]=}"
    assert z1[2] - z2[2] < tol, f"Step size mismatch: {z1[0]=} != {z2[0]=}"

    integrand = 2 * np.power(z1, 2 * r_power + 2) * wf1 * wf2
    if integration_method == "trapezoid":
        return float(scipy.integrate.trapezoid(integrand, x=z1))
    elif integration_method == "simpson":
        return float(scipy.integrate.simpson(integrand, x=z1))
    else:
        raise ValueError(f"Invalid integration method: {integration_method}")
