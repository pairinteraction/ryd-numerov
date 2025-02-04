from collections.abc import Sequence
from typing import Callable, Union

import numpy as np
from numba import njit


def _run_numerov_integration_python(
    x_start: float,
    x_stop: float,
    dx: float,
    y0: float,
    y1: float,
    g_list: Union[Sequence[float], np.ndarray],
    x_min: float,
) -> list[float]:
    r"""Run the Numerov integration algorithm.

    This means, run the Numerov method, which is defined for

    .. math::
        \frac{d^2}{dx^2} y(x) = - g(x) y(x)

    as

    .. math::
        y_{n+1} (1 + \frac{h^2}{12} g_{n+1}) = 2 y_n (1 - \frac{5 h^2}{12} g_n) - y_{n-1} (1 + \frac{h^2}{12} g_{n-1})

    Args:
        x_start: The initial value of the x-coordinate.
        x_stop: The final value of the x-coordinate.
        dx: The step size of the integration (can be negative).
        y0: The initial value of the function y(x) at the first (or last if run_backward) x-value.
        y1: The initial value of the function y(x) at the second (or second last if run_backward) x-value.
        g_list: A list of the values of the function g(x) at each x-value.
        x_min: The minimum value of the x-coordinate, until which the integration should be run.
            Once the x-value reaches x_min, we check if the function y(x) is zero and stop the integration.

    Returns:
        y_list: A list of the values of the function y(x) at each x-value

    """
    y_list = [y0, y1]

    i = 2
    x = x_start + 2 * dx
    sign = dx / abs(dx)

    while sign * x < sign * x_stop:
        new_y = (
            2 * (1 - 5 * dx**2 / 12 * g_list[i - 1]) * y_list[i - 1] - (1 + dx**2 / 12 * g_list[i - 2]) * y_list[i - 2]
        ) / (1 + dx**2 / 12 * g_list[i])

        # TODO this only works for positve y,
        # i.e. for backward integration always, but for forward integration only if there is an even number of nodes
        if (sign * x > sign * x_min) and y_list[-2] > y_list[-1] and y_list[-1] > 0:
            if new_y < 0:
                break
            if y_list[-1] < new_y:
                break

        y_list.append(new_y)
        x += dx
        i += 1

    return y_list


run_numerov_integration: Callable = njit(cache=True)(_run_numerov_integration_python)
