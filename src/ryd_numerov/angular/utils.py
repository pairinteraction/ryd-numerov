from __future__ import annotations

from functools import lru_cache, wraps
from typing import TYPE_CHECKING, Callable, TypeVar

import numpy as np
from sympy import Integer
from sympy.physics.wigner import (
    wigner_3j as sympy_wigner_3j,
    wigner_6j as sympy_wigner_6j,
    wigner_9j as sympy_wigner_9j,
)

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    P = ParamSpec("P")
    R = TypeVar("R")

    def lru_cache(maxsize: int) -> Callable[[Callable[P, R]], Callable[P, R]]: ...  # type: ignore [no-redef]


def sympify_args(func: Callable[P, R]) -> Callable[P, R]:
    """Check that quantum numbers are valid and convert to sympy.Integer (and half-integer)."""

    def check_arg(arg: float) -> Integer:
        if arg.is_integer():
            return Integer(int(arg))
        if (arg * 2).is_integer():
            return Integer(int(arg * 2)) / Integer(2)
        raise ValueError(f"Invalid input to {func.__name__}: {arg}.")

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        _args = [check_arg(arg) for arg in args]  # type: ignore[arg-type]
        _kwargs = {key: check_arg(value) for key, value in kwargs.items()}  # type: ignore[arg-type]
        return func(*_args, **_kwargs)

    return wrapper


def calc_wigner_3j(j1: float, j2: float, j3: float, m1: float, m2: float, m3: float) -> float:
    if not j1 <= j2 <= j3:  # better use of caching
        args_nd = np.array([j1, j2, j3, m1, m2, m3])
        inds = np.argsort(args_nd[:3])
        wigner = calc_wigner_3j(*args_nd[:3][inds], *args_nd[3:][inds])
        if (inds[1] - inds[0]) in [1, -2]:
            return wigner
        return minus_one_pow(j1 + j2 + j3) * wigner

    if m3 < 0 or (m3 == 0 and m2 < 0):  # better use of caching
        return minus_one_pow(j1 + j2 + j3) * calc_wigner_3j(j1, j2, j3, -m1, -m2, -m3)

    return _calc_wigner_3j(j1, j2, j3, m1, m2, m3)


@lru_cache(maxsize=10_000)
@sympify_args
def _calc_wigner_3j(j1: float, j2: float, j3: float, m1: float, m2: float, m3: float) -> float:
    return float(sympy_wigner_3j(j1, j2, j3, m1, m2, m3).evalf())


def calc_wigner_6j(j1: float, j2: float, j3: float, j4: float, j5: float, j6: float) -> float:
    if not j1 <= j4:  # better use of caching
        return calc_wigner_6j(j4, j2, j6, j1, j5, j3)

    if not j2 <= j5:  # better use of caching
        return calc_wigner_6j(j1, j5, j6, j4, j2, j3)

    if not j1 <= j2 <= j3:  # better use of caching
        args_nd = np.array([j1, j2, j3, j4, j5, j6])
        inds = np.argsort(args_nd[:3])
        return calc_wigner_6j(*args_nd[:3][inds], *args_nd[3:][inds])

    return _calc_wigner_6j(j1, j2, j3, j4, j5, j6)


@lru_cache(maxsize=10_000)
@sympify_args
def _calc_wigner_6j(j1: float, j2: float, j3: float, j4: float, j5: float, j6: float) -> float:
    return float(sympy_wigner_6j(j1, j2, j3, j4, j5, j6).evalf())


@lru_cache(maxsize=10_000)
@sympify_args
def calc_wigner_9j(
    j1: float, j2: float, j3: float, j4: float, j5: float, j6: float, j7: float, j8: float, j9: float
) -> float:
    return float(sympy_wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9).evalf())


def clebsch_gordan_6j(j1: float, j2: float, j3: float, j12: float, j23: float, j_tot: float) -> float:
    """Calculate the overlap between <((j1,j2)j12,j3)j_tot|(j1,(j2,j3)j23)j_tot>.

    We follow the convention of equation (6.1.5) from Edmonds 1985 "Angular Momentum in Quantum Mechanics".

    See Also:
        - https://en.wikipedia.org/wiki/Racah_W-coefficient
        - https://en.wikipedia.org/wiki/6-j_symbol

    Args:
        j1: Spin quantum number 1.
        j2: Spin quantum number 2.
        j3: Spin quantum number 3.
        j12: Total spin quantum number of j1 + j2.
        j23: Total spin quantum number of j2 + j3.
        j_tot: Total spin quantum number of j1 + j2 + j3.

    Returns:
        The Clebsch-Gordan coefficient <((j1,j2)j12,j3)j_tot|(j1,(j2,j3)j23)j_tot>.

    """
    prefactor: float = minus_one_pow(j1 + j2 + j3 + j_tot) * np.sqrt((2 * j12 + 1) * (2 * j23 + 1))
    wigner_6j = calc_wigner_6j(j1, j2, j12, j3, j_tot, j23)
    return prefactor * wigner_6j


def clebsch_gordan_9j(
    j1: float, j2: float, j12: float, j3: float, j4: float, j34: float, j13: float, j24: float, j_tot: float
) -> float:
    """Calculate the overlap between <((j1,j2)j12,(j3,j4)j34))j_tot|((j1,j3)j13,(j2,j4)j24))j_tot>.

    We follow the convention of equation (6.4.2) from Edmonds 1985 "Angular Momentum in Quantum Mechanics".

    See Also:
        - https://en.wikipedia.org/wiki/9-j_symbol

    Args:
        j1: Spin quantum number 1.
        j2: Spin quantum number 2.
        j12: Total spin quantum number of j1 + j2.
        j3: Spin quantum number 1.
        j4: Spin quantum number 2.
        j34: Total spin quantum number of j1 + j2.
        j13: Total spin quantum number of j1 + j3.
        j24: Total spin quantum number of j2 + j4.
        j_tot: Total spin quantum number of j1 + j2 + j3 + j4.

    Returns:
        The Clebsch-Gordan coefficient <((j1,j2)j12,(j3,j4)j34))j_tot|((j1,j3)j13,(j2,j4)j24))j_tot>.

    """
    prefactor: float = np.sqrt((2 * j12 + 1) * (2 * j34 + 1) * (2 * j13 + 1) * (2 * j24 + 1))
    return prefactor * calc_wigner_9j(j1, j2, j12, j3, j4, j34, j13, j24, j_tot)


def minus_one_pow(n: float) -> int:
    if n % 2 == 0:
        return 1
    if n % 2 == 1:
        return -1
    raise ValueError(f"Invalid input {n}.")


def check_triangular(j1: float, j2: float, j3: float) -> bool:
    return abs(j1 - j2) <= j3 <= j1 + j2
