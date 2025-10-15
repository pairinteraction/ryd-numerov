from functools import lru_cache

import numpy as np
from sympy import Integer
from sympy.physics.wigner import (
    wigner_3j as sympy_wigner_3j,
    wigner_6j as sympy_wigner_6j,
    wigner_9j as sympy_wigner_9j,
)


def _check_quantum_numbers(qns: list[float]) -> list[float]:
    """Check that quantum numbers are valid and convert to int or half-integer."""
    for i, qn in enumerate(qns):
        if qn % 1 == 0:
            qns[i] = int(qn)
        elif qn % 0.5 == 0:
            qns[i] = Integer(2 * qn) / Integer(2)
        else:
            raise ValueError(f"Invalid input {qn}.")
    return qns


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
def _calc_wigner_3j(j1: float, j2: float, j3: float, m1: float, m2: float, m3: float) -> float:
    qns = _check_quantum_numbers([j1, j2, j3, m1, m2, m3])
    return float(sympy_wigner_3j(*qns).evalf())


@lru_cache(maxsize=10_000)
def calc_wigner_6j(j1: float, j2: float, j3: float, j4: float, j5: float, j6: float) -> float:
    qns = _check_quantum_numbers([j1, j2, j3, j4, j5, j6])
    return float(sympy_wigner_6j(*qns).evalf())


@lru_cache(maxsize=10_000)
def calc_wigner_9j(
    j1: float, j2: float, j3: float, j4: float, j5: float, j6: float, j7: float, j8: float, j9: float
) -> float:
    qns = _check_quantum_numbers([j1, j2, j3, j4, j5, j6, j7, j8, j9])
    return float(sympy_wigner_9j(*qns).evalf())


def clebsch_gordan_6j(s1: float, s2: float, s_tot: float, l1: float, j1: float, j_tot: float) -> float:
    """Calculate the overlap between <((s1,s2)s_tot,l1)j_tot|((s1,l1)j1,s2)j_tot>.

    See Also:
    - https://en.wikipedia.org/wiki/Racah_W-coefficient
    - https://en.wikipedia.org/wiki/6-j_symbol

    Args:
        s1: Spin of electron 1.
        s2: Spin of electron 2.
        s_tot: Total spin of both electrons.
        l1: Orbital angular momentum of electron 1.
        j1: Total angular momentum of electron 1.
        j_tot: Total angular momentum of both electrons.

    Returns:
        The Clebsch-Gordan coefficient <((s1,s2)s_tot,l1)j_tot|((s1,l1)j1,s2)j_tot>.

    """
    racah_w = minus_one_pow(j_tot + l1 + s1 + s2) * calc_wigner_6j(j_tot, l1, s_tot, s1, s2, j1)
    prefactor: float = np.sqrt((2 * s_tot + 1) * (2 * j1 + 1))
    return prefactor * racah_w


def clebsch_gordan_9j(
    s1: float, s2: float, s_tot: float, l1: float, l2: float, l_tot: float, j1: float, j2: float, j_tot: float
) -> float:
    """Calculate the overlap between <((s1,s2)s_tot,(l1,l2)l_tot))j_tot|((s1,l1)j1,(s2,l2)j2))j_tot>.

    See Also:
    - https://en.wikipedia.org/wiki/9-j_symbol

    Args:
        s1: Spin of electron 1.
        s2: Spin of electron 2.
        s_tot: Total spin of both electrons.
        l1: Orbital angular momentum of electron 1.
        l2: Orbital angular momentum of electron 2.
        l_tot: Total orbital angular momentum of both electrons.
        j1: Total angular momentum of electron 1.
        j2: Total angular momentum of electron 2.
        j_tot: Total angular momentum of both electrons.

    Returns:
        The Clebsch-Gordan coefficient <((s1,s2)s_tot,(l1,l2)l_tot))j_tot|((s1,l1)j1,(s2,l2)j2))j_tot>.

    """
    prefactor: float = np.sqrt((2 * s_tot + 1) * (2 * l_tot + 1) * (2 * j1 + 1) * (2 * j2 + 1))
    return prefactor * calc_wigner_9j(s1, s2, s_tot, l1, l2, l_tot, j1, j2, j_tot)


def minus_one_pow(n: float) -> int:
    if n % 2 == 0:
        return 1
    if n % 2 == 1:
        return -1
    raise ValueError(f"Invalid input {n}.")


def check_triangular(j1: float, j2: float, j3: float) -> bool:
    return abs(j1 - j2) <= j3 <= j1 + j2
