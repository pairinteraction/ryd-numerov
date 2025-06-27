import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.integrate

if TYPE_CHECKING:
    from ryd_numerov.rydberg import RydbergState
    from ryd_numerov.units import NDArray

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
    if (
        k_radial == 0
        and (state1.l, state1.j) == (state2.l, state2.j)
        and state1.get_energy("a.u.") == state2.get_energy("a.u.")
    ):
        if state1.n == state2.n:
            return 1
        if isinstance(state1.n, int) and isinstance(state2.n, int):
            return 0

    # Ensure wavefunctions are integrated before accessing the grid
    wf1 = state1.wavefunction
    wf2 = state2.wavefunction
    return _calc_radial_matrix_element_from_w_z(
        wf1.grid.z_list, wf1.w_list, wf2.grid.z_list, wf2.w_list, k_radial, integration_method
    )


def _calc_radial_matrix_element_from_w_z(
    z1: "NDArray",
    w1: "NDArray",
    z2: "NDArray",
    w2: "NDArray",
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
    z_min = max(z1[0], z2[0])
    z_max = min(z1[-1], z2[-1])
    if z_max <= z_min:
        logger.debug("No overlapping grid points between states, returning 0")
        return 0

    # Select overlapping points
    dz = z1[1] - z1[0]
    if z1[0] < z_min - dz / 2:
        ind = int((z_min - z1[0]) / dz + 0.5)
        z1 = z1[ind:]
        w1 = w1[ind:]
    elif z2[0] < z_min - dz / 2:
        ind = int((z_min - z2[0]) / dz + 0.5)
        z2 = z2[ind:]
        w2 = w2[ind:]

    if z1[-1] > z_max + dz / 2:
        ind = int((z1[-1] - z_max) / dz + 0.5)
        z1 = z1[:-ind]
        w1 = w1[:-ind]
    elif z2[-1] > z_max + dz / 2:
        ind = int((z2[-1] - z_max) / dz + 0.5)
        z2 = z2[:-ind]
        w2 = w2[:-ind]

    _sanity_check_integration(z1, z2)

    integrand = 2 * w1 * w2
    integrand = _multiply_by_powers(integrand, z1, 2 * k_radial + 2)

    return _integrate(integrand, dz, integration_method)


def _multiply_by_powers(result: "NDArray", base: "NDArray", exponent: int) -> "NDArray":
    """Calculate result * base**(exponent) in an optimized way."""
    base_powers = {0: base}
    for i in range(exponent):
        if (exponent // 2**i) % 2 == 1:
            result *= base_powers[i]
            exponent -= 2**i
        if exponent == 0:
            break
        base_powers[i + 1] = np.square(base_powers[i])
    return result


def _sanity_check_integration(z1: "NDArray", z2: "NDArray") -> None:
    tol = 1e-10
    assert len(z1) == len(z2), f"Length mismatch: {len(z1)=} != {len(z2)=}"
    assert z1[0] - z2[0] < tol, f"First point mismatch: {z1[0]=} != {z2[0]=}"
    assert z1[1] - z2[1] < tol, f"Second point mismatch: {z1[1]=} != {z2[1]=}"
    assert z1[2] - z2[2] < tol, f"Third point mismatch: {z1[2]=} != {z2[2]=}"
    assert z1[-1] - z2[-1] < tol, f"Last point mismatch: {z1[-1]=} != {z2[-1]=}"


def _integrate(integrand: "NDArray", dz: float, method: INTEGRATION_METHODS) -> float:
    """Integrate the given integrand using the specified method."""
    if method == "sum":
        value = np.sum(integrand) * dz
    elif method == "trapezoid":
        value = np.trapezoid(integrand, dx=dz)
    elif method == "scipy_trapezoid":
        value = scipy.integrate.trapezoid(integrand, dx=dz)
    elif method == "scipy_simpson":
        value = scipy.integrate.simpson(integrand, dx=dz)
    else:
        raise ValueError(f"Invalid integration method: {method}")

    return float(value)
