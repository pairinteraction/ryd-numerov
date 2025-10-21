from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np

from ryd_numerov.angular.utils import calc_wigner_3j, calc_wigner_6j, minus_one_pow

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    P = ParamSpec("P")
    R = TypeVar("R")

AngularMomentumQuantumNumbers = Literal[
    "i_c", "s_c", "l_c", "s_r", "l_r", "s_tot", "l_tot", "j_c", "j_r", "j_tot", "f_c", "f_tot"
]
IdentityOperators = Literal[
    "identity_i_c",
    "identity_s_c",
    "identity_l_c",
    "identity_s_r",
    "identity_l_r",
    "identity_s_tot",
    "identity_l_tot",
    "identity_j_c",
    "identity_j_r",
    "identity_j_tot",
    "identity_f_c",
    "identity_f_tot",
]
AngularOperatorType = Literal[
    "SPHERICAL",
    AngularMomentumQuantumNumbers,
    IdentityOperators,
]


def calc_reduced_spherical_matrix_element(l_r_final: int, l_r_initial: int, kappa: int) -> float:
    r"""Calculate the reduced spherical matrix element (l_r_final || \hat{Y}_{k} || l_r_initial).

    We follow the convention of equation (5.4.5) from Edmonds 1985 "Angular Momentum in Quantum Mechanics".

    See Also:
        - https://en.wikipedia.org/wiki/3-j_symbol#Relation_to_spherical_harmonics;_Gaunt_coefficients

    The matrix elements of the spherical operators are given by:

    .. math::
        (l_r_final || \hat{Y}_{k} || l_r_initial)
            = (-1)^{l_r_final} \sqrt{(2 * l_r_final + 1)(2 * l_r_initial + 1)} * \sqrt{\frac{2 * \kappa + 1}{4 \pi}}
                                    \mathrm{Wigner3j}(l_r_final, k, l_r_initial; 0, 0, 0)

    Args:
        l_r_final: The orbital momentum quantum number of the final state.
        l_r_initial: The orbital momentum quantum number of the initial state.
        kappa: The quantum number :math:`\kappa` of the angular momentum operator.

    Returns:
        The reduced matrix element :math:`(l_r_final || \hat{Y}_{k} || l_r_initial)`.

    """
    prefactor = (
        minus_one_pow(l_r_final)
        * np.sqrt((2 * l_r_final + 1) * (2 * l_r_initial + 1))
        * np.sqrt((2 * kappa + 1) / (4 * np.pi))
    )
    wigner_3j = calc_wigner_3j(l_r_final, kappa, l_r_initial, 0, 0, 0)
    return prefactor * wigner_3j  # type: ignore [no-any-return]


def calc_reduced_spin_matrix_element(s_final: float, s_initial: float) -> float:
    r"""Calculate the reduced spin matrix element (s_final || \hat{s} || s_initial).

    The spin operator \hat{s} must be the operator corresponding to the quantum number s_final and s_initial.

    We follow the convention of equation (5.4.3) from Edmonds 1985 "Angular Momentum in Quantum Mechanics".
    The matrix elements of the spin operators are given by:

    .. math::
        (s_final || \hat{s} || s_initial)
            = \sqrt{(2 * s_final + 1) * (s_final + 1) * s_final} * \delta_{s_final, s_initial}

    Args:
        s_final: The spin quantum number of the final state.
        s_initial: The spin quantum number of the initial state.

    Returns:
        The reduced matrix element :math:`(s_final || \hat{s} || s_initial)`.

    """
    if s_final != s_initial:
        return 0
    return np.sqrt((2 * s_final + 1) * (s_final + 1) * s_final)  # type: ignore [no-any-return]


def calc_reduced_identity_matrix_element(s_final: float, s_initial: float) -> float:
    r"""Calculate the reduced identity matrix element (s_final || \id || s_initial).

    We follow the convention from Edmonds 1985 "Angular Momentum in Quantum Mechanics"
    (using equation (5.4.1) and (3.7.9)).
    The reduced matrix elements of the identity operator is given by:

    .. math::
        (s_final || \id || s_initial)
            = \sqrt{(2 * s_final + 1)} * \delta_{s_final, s_initial}

    Args:
        s_final: The spin quantum number of the final state.
        s_initial: The spin quantum number of the initial state.

    Returns:
        The reduced matrix element :math:`(s_final || \id || s_initial)`.

    """
    if s_final != s_initial:
        return 0
    return np.sqrt(2 * s_final + 1)  # type: ignore [no-any-return]


def calc_prefactor_of_operator_in_coupled_scheme(
    f1: float,
    f2: float,
    f12: float,
    i1: float,
    i2: float,
    i12: float,
    kappa: int,
    operator_acts_on: Literal["first", "second"],
) -> float:
    r"""Calculate the prefactor of the reduced matrix element for an operator acting on a state in a coupled scheme.

    Here we follow equation (7.1.7) and (7.1.8) from Edmonds 1985 "Angular Momentum in Quantum Mechanics".
    This means, if the operator only acts on the first quantum number (thus it must be f2 = i2),
    the reduced matrix element is given by

    .. math::
        \langle f1, f2, f12 || \hat{O}_{\kappa} || i1, i2, i12 \rangle
        = (-1)^{f1 + i2 + i12 + \kappa} * sqrt((2 * f12 + 1)(2 * i12 + 1))
            * \mathrm{Wigner6j}(f1, f12, i2; i12, i1, \kappa) * \langle f1 || \hat{O}_{\kappa} || i1 \rangle
        = prefactor  * \langle f1 || \hat{O}_{\kappa} || i1 \rangle

    and if the operator only acts on the second quantum number (thus it must be f1 = i1),
    the reduced matrix element is given by

    .. math::
        \langle f1, f2, f12 || \hat{O}_{\kappa} || i1, i2, i12 \rangle
        = (-1)^{i1 + i2 + f12 + \kappa} * sqrt((2 * f12 + 1)(2 * i12 + 1))
            * \mathrm{Wigner6j}(f2, f12, i1; i12, i2, \kappa) * \langle f2 || \hat{O}_{\kappa} || i2 \rangle
        = prefactor  * \langle f2 || \hat{O}_{\kappa} || i2 \rangle

    This function calculates and returns the prefactor.

    Args:
        f1: The quantum number of the first particle of the final state.
        f2: The quantum number of the second particle of the final state.
        f12: The total quantum number of the final state.
        i1: The quantum number of the first particle of the initial state.
        i2: The quantum number of the second particle of the initial state.
        i12: The total quantum number of the initial state.
        kappa: The rank :math:`\kappa` of the operator.
        operator_acts_on: Indicates on which particle the operator acts on (must be 'first' or 'second').

    """
    if operator_acts_on == "first":
        if f2 != i2:
            raise ValueError("If operator_acts_on first, f2 must be equal to i2.")
        return (  # type: ignore [no-any-return]
            minus_one_pow(f1 + i2 + i12 + kappa)
            * np.sqrt((2 * f12 + 1) * (2 * i12 + 1))
            * calc_wigner_6j(f1, f12, i2, i12, i1, kappa)
        )
    if operator_acts_on == "second":
        if f1 != i1:
            raise ValueError("If operator_acts_on second, f1 must be equal to i1.")
        return (  # type: ignore [no-any-return]
            minus_one_pow(i1 + i2 + f12 + kappa)
            * np.sqrt((2 * f12 + 1) * (2 * i12 + 1))
            * calc_wigner_6j(f2, f12, i1, i12, i2, kappa)
        )
    raise ValueError("operator_acts_on must be 'first' or 'second' in calc_prefactor_of_operator_in_coupled_scheme.")
