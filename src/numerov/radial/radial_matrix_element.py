import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.integrate

if TYPE_CHECKING:
    from numerov.rydberg import RydbergState

logger = logging.getLogger(__name__)

INTEGRATION_METHODS = Literal["sum", "trapezoid", "scipy_simpson", "scipy_trapezoid"]


def calc_radial_matrix_element(
    state1: "RydbergState",
    state2: "RydbergState",
    k_radial: int = 0,
    integration_method: INTEGRATION_METHODS = "sum",
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
        k_radial: Power of r in the matrix element
            (default=0, this corresponds to the overlap integral \int dr r^2 R_1(r) R_2(r))
        integration_method: Integration method to use, one of ["sum", "trapezoid", "scipy_simpson", "scipy_trapezoid"]
            (default="sum")

    Returns:
        float: The radial matrix element in atomic units.

    """
    # Special cases for the overlap integral (k_radial = 0)
    if k_radial == 0 and (state1.l, state1.j) == (state2.l, state2.j):
        if state1.n == state2.n:
            return 1
        else:
            return 0

    # Ensure wavefunctions are integrated before accessing the grid
    wf1 = state1.wavefunction
    wf2 = state2.wavefunction
    return _calc_radial_matrix_element_from_w_z(
        wf1.grid.zlist, wf1.wlist, wf2.grid.zlist, wf2.wlist, k_radial, integration_method
    )


def _calc_radial_matrix_element_from_w_z(
    z1: np.ndarray,
    w1: np.ndarray,
    z2: np.ndarray,
    w2: np.ndarray,
    k_radial: int = 0,
    integration_method: INTEGRATION_METHODS = "sum",
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
        k_radial: Power of r in the matrix element
            (default=0, this corresponds to the overlap integral \int dr r^2 R_1(r) R_2(r))
        integration_method: Integration method to use, one of ["sum", "trapezoid", "scipy_simpson", "scipy_trapezoid"]
            (default="sum")

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
    dz = z1[1] - z1[0]
    if z1[0] < zmin - dz / 2:
        ind = int((zmin - z1[0]) / dz + 0.5)
        z1 = z1[ind:]
        w1 = w1[ind:]
    elif z2[0] < zmin - dz / 2:
        ind = int((zmin - z2[0]) / dz + 0.5)
        z2 = z2[ind:]
        w2 = w2[ind:]

    if z1[-1] > zmax + dz / 2:
        ind = int((z1[-1] - zmax) / dz + 0.5)
        z1 = z1[:-ind]
        w1 = w1[:-ind]
    elif z2[-1] > zmax + dz / 2:
        ind = int((z2[-1] - zmax) / dz + 0.5)
        z2 = z2[:-ind]
        w2 = w2[:-ind]

    tol = 1e-10
    assert len(z1) == len(z2), f"Length mismatch: {len(z1)=} != {len(z2)=}"
    assert z1[0] - z2[0] < tol, f"First point mismatch: {z1[0]=} != {z2[0]=}"
    assert z1[1] - z2[1] < tol, f"Second point mismatch: {z1[1]=} != {z2[1]=}"
    assert z1[2] - z2[2] < tol, f"Third point mismatch: {z1[2]=} != {z2[2]=}"
    assert z1[-1] - z2[-1] < tol, f"Last point mismatch: {z1[-1]=} != {z2[-1]=}"

    integrand = 2 * w1 * w2
    for _ in range(2 * k_radial + 2):
        integrand *= z1

    if integration_method == "sum":
        return np.sum(integrand) * dz
    if integration_method == "trapezoid":
        return float(np.trapz(integrand, dx=dz))
    if integration_method == "scipy_trapezoid":
        return float(scipy.integrate.trapezoid(integrand, dx=dz))
    if integration_method == "scipy_simpson":
        return float(scipy.integrate.simpson(integrand, dx=dz))

    raise ValueError(f"Invalid integration method: {integration_method}")
