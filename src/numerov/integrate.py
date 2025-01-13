from typing import Literal

import numpy as np
from numba import njit


def integrate_hydrogen(
    energy: float,
    Z: float,
    n: int,
    l: int,
    dx: float,
    xmin: float,
    xmax: float,
    direction: Literal["forward", "backward"] = "backward",
    epsilon_u: float = 1e-10,
    use_njit: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Integrate the radial Schrödinger equation for the hydrogen atom using the Numerov method.

    We solve the radial dimensionless Schrödinger equation for the hydrogen atom

    .. math::
        \frac{d^2}{dx^2} u(x) = - \left[ \epsilon + \frac{2 Z}{x} - \frac{l(l+1)}{x^2} \right] u(x)

    using the Numerov method, see `run_numerov_integration`.

    Args:
        energy: The energy of the desired electronic state in dimensionless units.
        Z: The atomic number of the nucleus.
        n: The principal quantum number of the desired electronic state.
        l: The angular momentum quantum number of the desired electronic state.
        dx: The step size of the integration in dimensionless units (corresponds to h in the equation above).
        xmin: The minimum value of the radial coordinate in dimensionless units.
        xmax: The maximum value of the radial coordinate in dimensionless units.
        direction: The direction in which to integrate the radial Schrödinger equation (either "forward" or "backward").

    Returns:
        x_list: A numpy array of the values of the radial coordinate at which the wavefunction was evaluated.
        u_list: A numpy array of the values of the radial wavefunction at each value of x_list.

    """
    assert direction in ["forward", "backward"], f"Invalid direction: {direction}"
    assert xmin > 0, "The minimum value of the radial coordinate must be greater than zero."

    x_list = np.arange(xmin, xmax + dx, dx)

    u_list = np.zeros_like(x_list, dtype=float)

    if direction == "forward":
        u_list[1] = epsilon_u
    else:  # backward
        u_list[-2] = (-1) ** ((n - l + 1) % 2) * epsilon_u

    g_list = energy + 2 * Z / x_list - l * (l + 1) / x_list**2

    if use_njit:
        u_list = njit_run_numerov_integration(x_list, u_list, g_list, direction=direction)
    else:
        u_list = run_numerov_integration(x_list, u_list, g_list, direction=direction)

    # normalize the wavefunction, such that
    # \int_{0}^{\infty} x^2 |R(x)|^2 dx = \int_{0}^{\infty} |u(x)|^2 dx = 1
    norm = np.sqrt(np.sum(u_list**2) * dx)
    u_list /= norm

    return x_list, u_list


def run_numerov_integration(
    x_list: np.ndarray,
    y_list: np.ndarray,
    g_list: np.ndarray,
    direction: Literal["forward", "backward"] = "backward",
) -> np.ndarray:
    """Run the Numerov integration algorithm.

    This means, run the Numerov method, which is defined for

    .. math::
        \frac{d^2}{dx^2} y(x) = - g(x) y(x)

    as

    .. math::
        y_{n+1} (1 + \frac{h^2}{12} g_{n+1}) = 2 y_n (1 - \frac{5 h^2}{12} g_n) - y_{n-1} (1 + \frac{h^2}{12} g_{n-1})

    Args:
        x_list: A list of the x-values at which the function y(x) is evaluated.
        y_list: A list of the y-values, in which the initial values
            (y[0] and y[1] for forward direction and y[-1] and y[-2] for backward direction) are already set.
        g_list: A list of the values of the function g(x) at each x-value.
        direction: The direction in which to integrate the function (either "forward" or "backward").

    Returns:
        y_list: A numpy array of the values of the function y(x) at each x-value
    """
    dx = x_list[1] - x_list[0]
    assert np.allclose(np.diff(x_list), dx), "The values of x_list must be equally spaced."

    if direction == "forward":
        for i in range(2, len(x_list)):
            y_list[i] = (
                2 * (1 - 5 * dx**2 / 12 * g_list[i - 1]) * y_list[i - 1]
                - (1 + dx**2 / 12 * g_list[i - 2]) * y_list[i - 2]
            ) / (1 + dx**2 / 12 * g_list[i])
    elif direction == "backward":
        for i in range(len(x_list) - 3, -1, -1):
            y_list[i] = (
                2 * (1 - 5 * dx**2 / 12 * g_list[i + 1]) * y_list[i + 1]
                - (1 + dx**2 / 12 * g_list[i + 2]) * y_list[i + 2]
            ) / (1 + dx**2 / 12 * g_list[i])
    else:
        raise ValueError(f"Invalid direction: {direction}")

    return y_list


njit_run_numerov_integration = njit(run_numerov_integration)
