from functools import lru_cache

import numpy as np
from sympy import Integer
from sympy.physics.wigner import (
    wigner_3j as sympy_wigner_3j,
    wigner_6j as sympy_wigner_6j,
)

HALF = 1 / Integer(2)


@lru_cache(maxsize=10_000)
def calc_wigner_3j(j_1: float, j_2: float, j_3: float, m_1: float, m_2: float, m_3: float) -> float:
    args = [j_1, j_2, j_3, m_1, m_2, m_3]
    for i, arg in enumerate(args):
        if arg % 1 == 0:
            args[i] = int(arg)
        elif arg % 0.5 == 0:
            args[i] = Integer(2 * arg) * HALF
        else:
            raise ValueError(f"Invalid input {arg}.")
    return float(sympy_wigner_3j(*args).evalf())


@lru_cache(maxsize=10_000)
def calc_wigner_6j(j_1: float, j_2: float, j_3: float, j_4: float, j_5: float, j_6: float) -> float:
    args = [j_1, j_2, j_3, j_4, j_5, j_6]
    for i, arg in enumerate(args):
        if arg % 1 == 0:
            args[i] = int(arg)
        elif arg % 0.5 == 0:
            args[i] = Integer(2 * arg) * HALF
        else:
            raise ValueError(f"Invalid input {arg}.")
    return float(sympy_wigner_6j(*args).evalf())


def clebsch_gordan_6j(s1: float, s2: float, s_tot: int, l: int, j1: float, j_tot: int) -> float:
    """Calculate the overlap between <(l,(s1,s2)S)J|(s2,(s1,l)j1)J>.

    See Also:
    - https://en.wikipedia.org/wiki/Racah_W-coefficient
    - https://en.wikipedia.org/wiki/6-j_symbol

    Args:
        s1: Spin of the Rydberg electron.
        s2: Spin of the core electron.
        s_tot: Total spin.
        l: Orbital angular of the Rydberg electron.
        j1: Total angular momentum of the Rydberg electron.
        j_tot: Total angular momentum.

    Returns:
        The Clebsch-Gordan coefficient <(l,(s1,s2)S)J|(s2,(s1,l)j1)J>.

    """
    racah_w = minus_one_pow(j_tot + l + s1 + s2) * calc_wigner_6j(j_tot, l, s_tot, s1, s2, j1)
    prefactor: float = np.sqrt((2 * s_tot + 1) * (2 * j1 + 1))
    return prefactor * racah_w


def minus_one_pow(n: float) -> int:
    if n % 2 == 0:
        return 1
    if n % 2 == 1:
        return -1
    raise ValueError(f"Invalid input {n}.")


def check_triangular(j1: float, j2: float, j3: float) -> bool:
    return abs(j1 - j2) <= j3 <= j1 + j2
