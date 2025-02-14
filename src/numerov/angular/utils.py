from functools import lru_cache

from sympy import Integer
from sympy.physics.wigner import (
    wigner_3j as sympy_wigner_3j,
)
from sympy.physics.wigner import (
    wigner_6j as sympy_wigner_6j,
)

HALF = 1 / Integer(2)


@lru_cache(maxsize=2_000)
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


@lru_cache(maxsize=2_000)
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


def minus_one_pow(n: float) -> int:
    if n % 2 == 0:
        return 1
    if n % 2 == 1:
        return -1
    raise ValueError(f"Invalid input {n}.")


def check_triangular(j1: float, j2: float, j3: float) -> bool:
    return abs(j1 - j2) <= j3 <= j1 + j2
