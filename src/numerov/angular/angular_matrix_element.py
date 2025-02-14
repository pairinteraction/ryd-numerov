from typing import TYPE_CHECKING, Literal, Union

import numpy as np

from numerov.angular.utils import calc_wigner_3j, calc_wigner_6j, check_triangular, minus_one_pow

if TYPE_CHECKING:
    from numerov.rydberg import RydbergState

OperatorType = Literal["L", "S", "Y", "p", "mu"]


def calc_angular_matrix_element(
    state_i: "RydbergState",
    state_f: "RydbergState",
    operator: OperatorType,
    kappa: int,
    q: int,
) -> float:
    r"""Calculate the angular matrix element $\bra{state_f} \hat{O}_{kq} \ket{state_i}$.

    For the states $\bra{state_f} = \bra{l,s,j,m}$ and $\ket{state_i} = \ket{l',s',j',m'}$,
    the angular matrix elements of the angular momentum operators $\hat{O}_{kq}$ are given by

    .. math::
        \bra{state_f} \hat{O}_{kq} \ket{state_i}
        = \bra{l,s,j,m} \hat{O}_{kq} \ket{l',s',j',m'}
        = \langle j', m', k, q | j, m \rangle \langle j || \hat{O}_{k0} || j' \rangle / \sqrt{2j + 1}
        = (-1)^{j' - \kappa + m} wigner_3j(j', kappa, j, m', q, -m)
        \langle j || \hat{O}_{k0} || j' \rangle

    where we first used the Wigner-Eckhart theorem
    and then the Wigner 3-j symbol to express the Clebsch-Gordan coefficient.

    Args:
        state_i: The initial state $\ket{state_i} = \ket{l',s',j',m'}$.
        state_f: The final state $\bra{state_f} = \bra{l,s,j,m}$.
        operator: The angular momentum operator type $\hat{O}_{kq}$.
            Can be one of the following:
                - "L" for the orbital angular momentum operator,
                - "S" for the spin angular momentum operator,
                - "Y" for the spherical harmonics operator,
                - "p" for the spherical multipole operator.
                - "mu" for the magnetic moment operator.
        kappa: The quantum number $\kappa$ of the angular momentum operator.
        q: The quantum number $q$ of the angular momentum operator.

    Returns:
        The angular matrix element $\bra{state_f} \hat{O}_{kq} \ket{state_i}$.

    """
    prefactor = minus_one_pow(state_f.j - state_f.m)
    reduced_matrix_element = calc_reduced_angular_matrix_element(state_i, state_f, operator, kappa)
    wigner_3j = calc_wigner_3j(state_f.j, kappa, state_i.j, -state_f.m, q, state_i.m)
    return prefactor * reduced_matrix_element * wigner_3j


def calc_reduced_angular_matrix_element(
    state_i: "RydbergState",
    state_f: "RydbergState",
    operator: OperatorType,
    kappa: int,
    _lazy_evaluation: bool = True,
) -> float:
    r"""Calculate the reduced matrix element $\langle j || \hat{O}_{k0} || j' \rangle$.

    The reduced matrix elements $\langle j || \hat{O}_{k0} || j' \rangle$ for
    $\bra{j} = \bra{\gamma, s, l, j}$ and $\ket{j'} = \ket{\gamma', s', l', j'}$
    simplify for the special cases $s = s'$ or $l = l'$ to the following expressions:
    (see https://www.phys.ksu.edu/reu2015/danielkeylon/Wigner.pdf, and Edmonds: "Angular Momentum in Quantum Mechanics")

    For $s = s'$ (i.e. when \hat{O}_{k0} only acts on l), the reduced matrix element is given by
    .. math::
        \langle \gamma, s, l, j || \hat{O}_{k0} || \gamma', s, l', j' \rangle
        = (-1)**(s + l + j' + kappa) sqrt{2j + 1} sqrt{2j' + 1} wigner_6j(l, j, s, j', l', kappa)
        \langle \gamma, l || \hat{O}_{k0} || \gamma', l' \rangle

    And for $l = l'$ (i.e. when \hat{O}_{k0} only acts on s), the reduced matrix element is given by
    .. math::
        \langle \gamma, s, l, j || \hat{O}_{k0} || \gamma', s', l, j' \rangle
        = (-1)**(s + l + j' + kappa) sqrt{2j + 1} sqrt{2j' + 1} wigner_6j(s, j, l, j', s', kappa)
        \langle \gamma, s || \hat{O}_{k0} || \gamma', s' \rangle


    Args:
        state_i: The initial state $\ket{state_i} = \ket{l',s',j',m'}$.
        state_f: The final state $\bra{state_f} = \bra{l,s,j,m}$.
        operator: The angular momentum operator $\hat{O}_{kq}$.
        kappa: The quantum number $\kappa$ of the angular momentum operator.

    Returns:
        The reduced matrix element $\langle j || \hat{O}_{k0} || j' \rangle$.

    """
    assert operator in ["L", "S", "Y", "p", "mu"]

    should_be_zero = False
    if not check_triangular(state_f.j, state_i.j, kappa):
        should_be_zero = True
    elif operator in ["Y", "p", "L"] and not check_triangular(state_f.l, state_i.l, kappa):
        should_be_zero = True
    elif operator in ["S"] and not check_triangular(state_f.s, state_i.s, kappa):
        should_be_zero = True
    elif operator in ["p", "Y"] and (state_f.l + state_i.l + kappa) % 2 != 0:
        should_be_zero = True
    elif operator in ["L", "S", "mu"] and (state_f.l != state_i.l or state_f.s != state_i.s):
        should_be_zero = True
    elif (operator == "S" and state_f.s == 0) or (operator == "L" and state_f.l == 0):
        should_be_zero = True

    if should_be_zero and _lazy_evaluation:
        return 0

    if operator == "mu":
        mu_B = 0.5  # Bohr magneton in atomic units
        g_s = 2.0023192
        value_s = calc_reduced_angular_matrix_element(state_i, state_f, "S", kappa)
        g_l = 1
        value_l = calc_reduced_angular_matrix_element(state_i, state_f, "L", kappa)
        mu = mu_B * (
            g_s * value_s + g_l * value_l
        )  # TODO note the missing minus is convention how we use it in pairinteraction ...
        return mu

    prefactor = np.sqrt(2 * state_i.j + 1) * np.sqrt(2 * state_f.j + 1)

    if operator == "S":
        prefactor *= minus_one_pow(state_f.l + state_i.s + state_f.j + kappa)
        if state_i.l != state_f.l:
            reduced_matrix_element = 0
        else:
            reduced_matrix_element = _calc_reduced_momentum_matrix_element(state_i.s, state_f.s, kappa)
        wigner_6j = calc_wigner_6j(state_f.s, state_f.j, state_f.l, state_i.j, state_i.s, kappa)
    else:
        prefactor *= minus_one_pow(state_f.l + state_f.s + state_i.j + kappa)
        if operator == "L":
            if state_i.s != state_f.s:
                reduced_matrix_element = 0
            else:
                reduced_matrix_element = _calc_reduced_momentum_matrix_element(state_i.l, state_f.l, kappa)
        else:
            reduced_matrix_element = _calc_reduced_multipole_matrix_element(state_i.l, state_f.l, operator, kappa)
        wigner_6j = calc_wigner_6j(state_f.l, state_f.j, state_f.s, state_i.j, state_i.l, kappa)

    value = prefactor * reduced_matrix_element * wigner_6j

    # Check that we catched all cases where the reduced matrix element is zero before
    if should_be_zero and value != 0:
        raise ValueError(
            f"The reduced angular matrix element for {state_i}, {state_f}, {operator}, {kappa} "
            "is not zero (but should be zero)."
        )
    if value == 0 and not should_be_zero:
        raise ValueError(
            f"The reduced angular matrix element for {state_i}, {state_f}, {operator}, {kappa}"
            "is zero (but should not be zero)."
        )

    return value


def _calc_reduced_momentum_matrix_element(j_i: Union[int, float], j_f: Union[int, float], kappa: int) -> float:
    r"""Calculate the reduced matrix element $(j_f||\hat{j}_{10}||j_i)$ for a momentum operator.

    The matrix elements of the momentum operators $j \in \{l, s\}$ are given by

    .. math::
        (j_f||\hat{j}_{10}||j_i) = \delta_{j_f, j_i} \sqrt{j_i(j_i+1)(2j_i+1)}

    Args:
        j_i: The angular momentum quantum number $j_i$ of the initial state.
        j_f: The angular momentum quantum number $j_f$ of the final state.
        kappa: The quantum number $\kappa$ of the angular momentum operator.

    Returns:
        The reduced matrix element $(j_f||\hat{j}_{10}||j_i)$.

    """
    if j_i != j_f or j_i == 0:
        return 0
    if kappa == 1:
        return np.sqrt(j_i * (j_i + 1) * (2 * j_i + 1))
    raise NotImplementedError("Currently only kappa=1 is supported.")


def _calc_reduced_multipole_matrix_element(l_i: int, l_f: int, operator: OperatorType, kappa: int) -> float:
    r"""Calculate the reduced matrix element $(l||\hat{p}_{k0}||l')$ for the multipole operator.

    The matrix elements of the multipole operators are given by (see also: Gaunt coefficient)

    .. math::
        (l||\hat{p}_{k0}||l') = (-1)^l \sqrt{(2l+1)(2l'+1)} \begin{pmatrix} l & k & l' \\ 0 & 0 & 0 \end{pmatrix}

    Args:
        l_i: The angular momentum quantum number $l$ of the initial state.
        l_f: The angular momentum quantum number $l'$ of the final state.
        operator: The multipole operator, either "Y" or "p".
        kappa: The quantum number $\kappa$ of the angular momentum operator.

    Returns:
        The reduced matrix element $(l||\hat{p}_{k0}||l')$.

    """
    assert operator in ["Y", "p"]

    prefactor = minus_one_pow(l_f)
    prefactor *= np.sqrt((2 * l_i + 1) * (2 * l_f + 1))
    if operator == "Y":
        prefactor *= np.sqrt((2 * kappa + 1) / (4 * np.pi))

    wigner_3j = calc_wigner_3j(l_f, kappa, l_i, 0, 0, 0)
    return prefactor * wigner_3j
