from collections.abc import Sequence
from typing import Union

import numpy as np
from numba import njit


@njit(cache=True)
def run_numerov_integration(
    dx: float,
    steps: int,
    y0: float,
    y1: float,
    g_list: Union[Sequence[float], np.ndarray],
    run_backward: bool = True,
) -> np.ndarray:
    r"""Run the Numerov integration algorithm.

    This means, run the Numerov method, which is defined for

    .. math::
        \frac{d^2}{dx^2} y(x) = - g(x) y(x)

    as

    .. math::
        y_{n+1} (1 + \frac{h^2}{12} g_{n+1}) = 2 y_n (1 - \frac{5 h^2}{12} g_n) - y_{n-1} (1 + \frac{h^2}{12} g_{n-1})

    Args:
        dx: The step size of the integration.
        steps: The number of steps to integrate.
        y0: The initial value of the function y(x) at the first (or last if run_backward) x-value.
        y1: The initial value of the function y(x) at the second (or second last if run_backward) x-value.
        g_list: A list of the values of the function g(x) at each x-value.
        run_backward (default: True): Whether to run the integration in the backward direction.

    Returns:
        y_list: A numpy array of the values of the function y(x) at each x-value

    """
    y_list = np.zeros(steps, dtype=float)
    if run_backward:
        y_list[-1] = y0
        y_list[-2] = y1
        for i in range(steps - 3, -1, -1):
            y_list[i] = (
                2 * (1 - 5 * dx**2 / 12 * g_list[i + 1]) * y_list[i + 1]
                - (1 + dx**2 / 12 * g_list[i + 2]) * y_list[i + 2]
            ) / (1 + dx**2 / 12 * g_list[i])
    else:  # forward
        y_list[0] = y0
        y_list[1] = y1
        for i in range(2, steps):
            y_list[i] = (
                2 * (1 - 5 * dx**2 / 12 * g_list[i - 1]) * y_list[i - 1]
                - (1 + dx**2 / 12 * g_list[i - 2]) * y_list[i - 2]
            ) / (1 + dx**2 / 12 * g_list[i])
    return y_list


def _python_run_numerov_integration(
    dx: float,
    steps: int,
    y0: float,
    y1: float,
    g_list: np.ndarray,
    run_backward: bool = True,
) -> np.ndarray:
    """Just a copy of the njit version above and only used for benchmarks."""
    y_list = np.zeros(steps, dtype=float)
    if run_backward:
        y_list[-1] = y0
        y_list[-2] = y1
        for i in range(steps - 3, -1, -1):
            y_list[i] = (
                2 * (1 - 5 * dx**2 / 12 * g_list[i + 1]) * y_list[i + 1]
                - (1 + dx**2 / 12 * g_list[i + 2]) * y_list[i + 2]
            ) / (1 + dx**2 / 12 * g_list[i])
    else:  # forward
        y_list[0] = y0
        y_list[1] = y1
        for i in range(2, steps):
            y_list[i] = (
                2 * (1 - 5 * dx**2 / 12 * g_list[i - 1]) * y_list[i - 1]
                - (1 + dx**2 / 12 * g_list[i - 2]) * y_list[i - 2]
            ) / (1 + dx**2 / 12 * g_list[i])
    return y_list
