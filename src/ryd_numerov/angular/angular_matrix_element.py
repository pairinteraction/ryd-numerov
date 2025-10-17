from functools import lru_cache
from typing import get_args

import numpy as np

from ryd_numerov.angular.utils import calc_wigner_3j, calc_wigner_6j, check_triangular, minus_one_pow
from ryd_numerov.units import OperatorType


@lru_cache(maxsize=10_000)
def calc_reduced_angular_matrix_element(
    s1: float,
    l1: int,
    j1: float,
    s2: float,
    l2: int,
    j2: float,
    operator: OperatorType,
    kappa: int,
    _lazy_evaluation: bool = True,
) -> float:
    r"""Calculate the reduced matrix element :math:`\langle j2 || \hat{O}_{k0} || j1 \rangle`.

    The reduced matrix elements :math:`\langle j2 || \hat{O}_{k0} || j1 \rangle` for
    :math:`\bra{j2} = \bra{\gamma_2, s2, l2, j2}` and :math:`\ket{j1} = \ket{\gamma_1, s1, l1, j1}`
    simplify for the special cases :math:`s2 = s1` or :math:`l2 = l1` to the following expressions:
    (see https://www.phys.ksu.edu/reu2015/danielkeylon/Wigner.pdf, and Edmonds: "Angular Momentum in Quantum Mechanics")

    For :math:`s2 = s1` (i.e. when :math:`\hat{O}_{k0}` only acts on :math:`l`), the reduced matrix element is given by

    .. math::
        \langle \gamma_{2}, s2, l2, j2 || \hat{O}_{k0} || \gamma_{1}, s2, l1, j1 \rangle
        = (-1)^{s2 + l2 + j1 + \kappa} \sqrt{2j + 1} \sqrt{2j1 + 1} \mathrm{Wigner6j}(l2, j2, s2, j1, l1, \kappa)
        \langle \gamma_{2}, l2 || \hat{O}_{k0} || \gamma_{1}, l1 \rangle

    And for :math:`l2 = l1` (i.e. when :math:`\hat{O}_{k0}` only acts on :math:`s`),
    the reduced matrix element is given by

    .. math::
        \langle \gamma_{2}, s2, l2, j2 || \hat{O}_{k0} || \gamma_{1}, s1, l2, j1 \rangle
        = (-1)^{s2 + l2 + j1 + \kappa} \sqrt{2j + 1} \sqrt{2 * j1 + 1} \mathrm{Wigner6j}(s2, j2, l2, j1, s1, \kappa)
        \langle \gamma_{2}, s2 || \hat{O}_{k0} || \gamma_{1}, s1 \rangle

    Note we changed the formulas to match the pairinteraction paper convention:
    https://doi.org/10.1088/1361-6455/aa743a

    Args:
        s1: The spin quantum number of the initial state.
        l1: The orbital quantum number of the initial state.
        j1: The total angular momentum quantum number of the initial state.
        s2: The spin quantum number of the final state.
        l2: The orbital quantum number of the final state.
        j2: The total angular momentum quantum number of the final state.
        operator: The operator type :math:`\hat{O}_{kq}` for which to calculate the matrix element.
            Can be one of "MAGNETIC", "ELECTRIC", "SPHERICAL".
        kappa: The quantum number :math:`\kappa` of the angular momentum operator.

    Returns:
        The reduced matrix element :math:`\langle j2 || \hat{O}_{k0} || j1 \rangle`.

    """
    assert operator in get_args(OperatorType), (
        f"Operator {operator} not supported, must be one of {get_args(OperatorType)}"
    )

    should_be_zero = check_reduced_angular_matrix_element_should_be_zero(s1, l1, j1, s2, l2, j2, operator, kappa)
    if _lazy_evaluation and should_be_zero:
        return 0

    if operator == "MAGNETIC":
        g_s = 2.0023192
        value_s = calc_reduced_angular_matrix_element(s1, l1, j1, s2, l2, j2, "MAGNETIC_S", kappa)
        g_l = 1
        value_l = calc_reduced_angular_matrix_element(s1, l1, j1, s2, l2, j2, "MAGNETIC_L", kappa)
        return g_s * value_s + g_l * value_l

    prefactor: float = np.sqrt(2 * j1 + 1) * np.sqrt(2 * j2 + 1)

    reduced_matrix_element: float
    if operator == "MAGNETIC_S":
        prefactor *= minus_one_pow(l2 + s1 + j2 + kappa)
        reduced_matrix_element = spin_like_matrix_element(s1, s2, kappa)
        wigner_6j = calc_wigner_6j(s2, j2, l2, j1, s1, kappa)
    else:
        prefactor *= minus_one_pow(l2 + s2 + j1 + kappa)
        if operator == "MAGNETIC_L":
            reduced_matrix_element = spin_like_matrix_element(l1, l2, kappa)
        else:
            reduced_matrix_element = spherical_like_matrix_element(l1, l2, operator, kappa)
        wigner_6j = calc_wigner_6j(l2, j2, s2, j1, l1, kappa)

    value = prefactor * reduced_matrix_element * wigner_6j

    # Check that we catched all cases where the reduced matrix element is zero before
    if should_be_zero and value != 0:
        raise ValueError(
            f"The reduced angular matrix element for {(s1, l1, j1)}, {(s2, l2, j2)}, {operator}, {kappa} "
            "is not zero (but should be zero)."
        )
    if value == 0 and not should_be_zero:
        raise ValueError(
            f"The reduced angular matrix element for {(s1, l1, j1)}, {(s2, l2, j2)}, {operator}, {kappa} "
            "is zero (but should not be zero)."
        )

    return value


def check_reduced_angular_matrix_element_should_be_zero(  # noqa: PLR0911
    s1: float, l1: int, j1: float, s2: float, l2: int, j2: float, operator: OperatorType, kappa: int
) -> bool:
    if not check_triangular(j2, j1, kappa):
        return True
    if operator in ["SPHERICAL", "ELECTRIC", "MAGNETIC_L"] and not check_triangular(l2, l1, kappa):
        return True
    if operator in ["SPHERICAL", "ELECTRIC"] and (l2 + l1 + kappa) % 2 != 0:
        return True
    if operator in ["MAGNETIC_L", "MAGNETIC_S", "MAGNETIC"] and l2 != l1:
        return True
    if (operator == "MAGNETIC_S" and s2 == 0) or (operator == "MAGNETIC_L" and l2 == 0):
        return True
    if s2 != s1:  # noqa: SIM103
        return True
    return False


def spin_like_matrix_element(x1: float, x2: float, kappa: int) -> float:
    r"""Calculate the reduced spin-like matrix element $(x2||\hat{x}_{10}||x1)$ for a spin-like operator.

    The matrix elements of the spin-like operators $x \in \{l, s\}$ are given by

    .. math::
        (x2||\hat{x}_{10}||x1) = \delta_{x2, x1} \sqrt{x1 (x1 + 1) (2 * x1 + 1)}

    Args:
        x1: The spin-like quantum number of the initial state.
        x2: The spin-like quantum number of the final state.
        kappa: The quantum number $\kappa$ of the spin-like operator.

    Returns:
        The reduced matrix element $(x2||\hat{x}_{10}||x1)$.

    """
    if kappa != 1:
        raise NotImplementedError("Currently only kappa=1 is supported.")

    if x1 != x2 or x1 == 0:
        return 0
    value: float = np.sqrt(x1 * (x1 + 1) * (2 * x1 + 1))
    return value


@lru_cache(maxsize=1_000)
def spherical_like_matrix_element(l1: int, l2: int, operator: OperatorType, kappa: int) -> float:
    r"""Calculate the reduced spherical-like matrix element $(l2||\hat{Y}_{k0}||l1)$.

    The matrix elements of the spherical operators are given by (see also: Gaunt coefficient)

    .. math::
        (l2||\hat{Y}_{k0}||l1) = (-1)^l2 \sqrt{(2 * l2 + 1)(2 * l1 + 1)} * \sqrt{\frac{2 * \kappa + 1}{4 \pi}}
                                    \begin{pmatrix} l2 & k & l1 \\ 0 & 0 & 0 \end{pmatrix}

    If the operator is the electric multipole operator, the prefactor $\sqrt{(2 * \kappa + 1) / (4 \pi)}$ is dropped.

    Args:
        l1: The orbital momentum quantum number of the initial state.
        l2: The orbital momentum quantum number of the final state.
        operator: The multipole operator, either "SPHERICAL" or "ELECTRIC".
        kappa: The quantum number $\kappa$ of the angular momentum operator.

    Returns:
        The reduced matrix element $(l2||\hat{Y}_{k0}||l1)$.

    """
    assert operator in ["SPHERICAL", "ELECTRIC"]

    prefactor: float = minus_one_pow(l2)
    prefactor *= np.sqrt((2 * l1 + 1) * (2 * l2 + 1))
    if operator == "SPHERICAL":
        prefactor *= np.sqrt((2 * kappa + 1) / (4 * np.pi))
    wigner_3j = calc_wigner_3j(l2, kappa, l1, 0, 0, 0)
    return prefactor * wigner_3j


def calc_prefactor_of_operator_in_coupled_scheme(
    f1: float, f2: float, f12: float, i1: float, i2: float, i12: float, kappa: int
) -> float:
    r"""Calculate the prefactor of the reduced matrix element for an operator acting on a state in a coupled scheme.

    Here we follow equation (7.1.7) from Edmonds 1985 "Angular Momentum in Quantum Mechanics".
    This means, for f2 = i2 (i.e. the operator only acts on the first quantum number),
    the reduced matrix element is given by

    .. math::
        \langle f1, f2, f12 || \hat{O}_{\kappa} || i1, i2, i12 \rangle
        = (-1)^{f1 + i2 + i12 + \kappa} * sqrt((2 * f12 + 1)(2 * i12 + 1))
            * \mathrm{Wigner6j}(f1, f12, i2; i12, i1, \kappa) * \langle f1 || \hat{O}_{\kappa} || i1 \rangle
        = prefactor  * \langle f1 || \hat{O}_{\kappa} || i1 \rangle

    This function calculates and returns the prefactor.

    Args:
        f1: The quantum number of the first particle of the final state.
        f2: The quantum number of the second particle of the final state.
        f12: The total quantum number of the final state.
        i1: The quantum number of the first particle of the initial state.
        i2: The quantum number of the second particle of the initial state.
        i12: The total quantum number of the initial state.
        kappa: The rank :math:`\kappa` of the operator.

    """
    if f2 != i2:
        raise ValueError("calc_prefactor_of_operator_in_coupled_scheme is meant to be used for f2 == i2 only.")
    return (  # type: ignore [no-any-return]
        minus_one_pow(f1 + i2 + i12 + kappa)
        * np.sqrt((2 * f12 + 1) * (2 * i12 + 1))
        * calc_wigner_6j(f1, f12, i2, i12, i1, kappa)
    )
