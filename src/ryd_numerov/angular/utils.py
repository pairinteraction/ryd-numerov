from __future__ import annotations

import math
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


# global variables to possibly improve the performance of wigner j calculations
# in the public release we will always use CHECK_ARGS = True and USE_SYMMETRIES = False to reduce potential of bugs
CHECK_ARGS = True
USE_SYMMETRIES = False


def sympify_args(func: Callable[P, R]) -> Callable[P, R]:
    """Check that quantum numbers are valid and convert to sympy.Integer (and half-integer)."""
    if not CHECK_ARGS:
        return func

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


@lru_cache(maxsize=10_000)
@sympify_args
def calc_wigner_3j(j1: float, j2: float, j3: float, m1: float, m2: float, m3: float) -> float:
    """Calculate the Wigner 3j symbol using lru_cache to improve performance."""
    return float(sympy_wigner_3j(j1, j2, j3, m1, m2, m3).evalf())


@lru_cache(maxsize=100_000)
@sympify_args
def calc_wigner_6j(j1: float, j2: float, j3: float, j4: float, j5: float, j6: float) -> float:
    """Calculate the Wigner 6j symbol using lru_cache to improve performance."""
    return float(sympy_wigner_6j(j1, j2, j3, j4, j5, j6).evalf())


@lru_cache(maxsize=10_000)
@sympify_args
def calc_wigner_9j(
    j1: float, j2: float, j3: float, j4: float, j5: float, j6: float, j7: float, j8: float, j9: float
) -> float:
    """Calculate the Wigner 9j symbol using lru_cache to improve performance."""
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
    prefactor = minus_one_pow(j1 + j2 + j3 + j_tot) * math.sqrt((2 * j12 + 1) * (2 * j23 + 1))
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
    prefactor = math.sqrt((2 * j12 + 1) * (2 * j34 + 1) * (2 * j13 + 1) * (2 * j24 + 1))
    return prefactor * calc_wigner_9j(j1, j2, j12, j3, j4, j34, j13, j24, j_tot)


def calc_wigner_3j_with_symmetries(j1: float, j2: float, j3: float, m1: float, m2: float, m3: float) -> float:
    """Calculate the Wigner 3j symbol using symmetries to reduce the number of symbols, that are not cached."""
    symmetry_factor: float = 1

    # even permutation -> sort smallest j to be j1
    if j2 < j1 and j2 < j3:
        j1, j2, j3, m1, m2, m3 = j2, j3, j1, m2, m3, m1
    elif j3 < j1 and j3 < j2:
        j1, j2, j3, m1, m2, m3 = j3, j1, j2, m3, m1, m2

    # odd permutation -> sort seccond smallest j to be j2
    if j3 < j2:
        symmetry_factor *= minus_one_pow(j1 + j2 + j3)
        j1, j2, j3, m1, m2, m3 = j1, j3, j2, m1, m3, m2  # noqa: PLW0127

    # sign of m -> make m1 positive (or m2 if m1==0)
    if m1 <= 0 or (m1 == 0 and m2 < 0):
        symmetry_factor *= minus_one_pow(j1 + j2 + j3)
        m1, m2, m3 = -m1, -m2, -m3

    # TODO Regge symmetries

    return symmetry_factor * calc_wigner_3j(j1, j2, j3, m1, m2, m3)


def calc_wigner_6j_with_symmetries(j1: float, j2: float, j3: float, j4: float, j5: float, j6: float) -> float:
    """Calculate the Wigner 6j symbol using symmetries to reduce the number of symbols, that are not cached."""
    # interchange upper and lower for 2 columns -> make j1 < j4 and j2 < j5
    if j4 < j1:
        j1, j2, j3, j4, j5, j6 = j4, j2, j6, j1, j5, j3  # noqa: PLW0127
    if j5 < j2:
        j1, j2, j3, j4, j5, j6 = j1, j5, j6, j4, j2, j3  # noqa: PLW0127

    # any permutation of columns -> make j1 <= j2 <= j3
    if j2 < j1 and j2 < j3:
        j1, j2, j3, j4, j5, j6 = j2, j1, j3, j5, j4, j6  # noqa: PLW0127
    elif j3 < j1 and j3 < j2:
        j1, j2, j3, j4, j5, j6 = j3, j2, j1, j6, j5, j4  # noqa: PLW0127

    if j3 < j2:
        j1, j2, j3, j4, j5, j6 = j1, j3, j2, j4, j6, j5  # noqa: PLW0127

    return calc_wigner_6j(j1, j2, j3, j4, j5, j6)


def calc_wigner_9j_with_symmetries(
    j1: float, j2: float, j3: float, j4: float, j5: float, j6: float, j7: float, j8: float, j9: float
) -> float:
    """Calculate the Wigner 9j symbol using symmetries to reduce the number of symbols, that are not cached."""
    symmetry_factor: float = 1
    js = [j1, j2, j3, j4, j5, j6, j7, j8, j9]

    # even permutation of rows and columns -> make smallest j to be j1
    min_j = min(js)
    if min_j not in js[:3]:
        if min_j in js[3:6]:
            js = [*js[3:6], *js[6:9], *js[0:3]]
        elif min_j in js[6:9]:
            js = [*js[6:9], *js[0:3], *js[3:6]]
    if js[0] != min_j:
        if js[1] == min_j:
            js = [js[1], js[2], js[0], js[4], js[5], js[3], js[7], js[8], js[6]]
        elif js[2] == min_j:
            js = [js[2], js[0], js[1], js[5], js[3], js[4], js[8], js[6], js[7]]

    # odd permutations of rows and columns-> make j2 <= j3 and j4 <= j7
    if js[2] < js[1]:
        symmetry_factor *= minus_one_pow(sum(js))
        js = [js[0], js[2], js[1], js[3], js[5], js[4], js[6], js[8], js[7]]
    if js[6] < js[3]:
        symmetry_factor *= minus_one_pow(sum(js))
        js = [*js[0:3], *js[6:9], *js[3:6]]

    # reflection about diagonal -> make j2 <= j4
    if js[3] < js[1]:
        js = [js[0], js[3], js[6], js[1], js[4], js[7], js[2], js[5], js[8]]

    return symmetry_factor * calc_wigner_9j(*js)


if USE_SYMMETRIES:
    calc_wigner_3j = calc_wigner_3j_with_symmetries  # type: ignore [assignment]
    calc_wigner_6j = calc_wigner_6j_with_symmetries  # type: ignore [assignment]
    calc_wigner_9j = calc_wigner_9j_with_symmetries  # type: ignore [assignment]


def minus_one_pow(n: float) -> int:
    """Calculate (-1)^n for an integer n and raise an error if n is not an integer."""
    if n % 2 == 0:
        return 1
    if n % 2 == 1:
        return -1
    raise ValueError(f"minus_one_pow: Invalid input {n=} is not an integer.")


def try_trivial_spin_addition(s_1: float, s_2: float, s_tot: float | None, name: str) -> float:
    """Try to determine s_tot from s_1 and s_2 if it is not given.

    If s_tot is None and cannot be uniquely determined from s_1 and s_2, raise an error.
    Otherwise return s_tot or the trivial sum s_1 + s_2.
    """
    if s_tot is None:
        if s_1 != 0 and s_2 != 0:
            msg = f"{name} must be set if both parts ({s_1} and {s_2}) are non-zero."
            raise ValueError(msg)
        s_tot = s_1 + s_2
    return float(s_tot)


def check_spin_addition_rule(s_1: float, s_2: float, s_tot: float) -> bool:
    """Check if the spin addition rule is satisfied.

    This means check the following conditions:
    - |s_1 - s_2| <= s_tot <= s_1 + s_2
    - s_1 + s_2 + s_tot is an integer
    """
    return abs(s_1 - s_2) <= s_tot <= s_1 + s_2 and (s_1 + s_2 + s_tot) % 1 == 0


def get_possible_quantum_number_values(s_1: float, s_2: float, s_tot: float | None) -> list[float]:
    """Determine a list of possible s_tot values from s_1 and s_2 if s_tot is not given, else return [s_tot]."""
    if s_tot is not None:
        return [s_tot]
    return [float(s) for s in np.arange(abs(s_1 - s_2), s_1 + s_2 + 1, 1)]
