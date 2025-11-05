from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import numpy as np
import pytest
from ryd_numerov.angular import AngularKetFJ, AngularKetJJ, AngularKetLS
from ryd_numerov.angular.angular_matrix_element import AngularMomentumQuantumNumbers

if TYPE_CHECKING:
    from ryd_numerov.angular.angular_ket import AngularKetBase, CouplingScheme
    from ryd_numerov.angular.angular_matrix_element import AngularOperatorType

TEST_KET_PAIRS = [
    (
        AngularKetLS(s_tot=1, l_r=1, j_tot=1, f_tot=1.5, species="Yb173"),
        AngularKetLS(s_tot=1, l_r=1, j_tot=1, f_tot=1.5, species="Yb173"),
    ),
    (
        AngularKetLS(s_tot=1, l_r=1, j_tot=1, f_tot=1.5, species="Yb173"),
        AngularKetFJ(f_c=2, l_r=1, j_r=1.5, f_tot=2.5, species="Yb173"),
    ),
    (
        AngularKetFJ(f_c=2, l_r=1, j_r=1.5, f_tot=2.5, species="Yb173"),
        AngularKetFJ(f_c=2, l_r=1, j_r=1.5, f_tot=2.5, species="Yb173"),
    ),
    (
        AngularKetFJ(i_c=2.5, s_c=0.5, l_c=0, s_r=0.5, l_r=1, j_c=0.5, f_c=2.0, j_r=1.5, f_tot=2.5),
        AngularKetFJ(i_c=2.5, s_c=0.5, l_c=0, s_r=0.5, l_r=2, j_c=0.5, f_c=2.0, j_r=1.5, f_tot=2.5),
    ),
]

TEST_KETS = [
    AngularKetLS(s_tot=1, l_r=1, j_tot=1, f_tot=1.5, species="Yb173"),
    AngularKetLS(s_tot=1, l_r=1, j_tot=1, f_tot=1.5, species="Yb173"),
    AngularKetJJ(l_r=1, j_r=1.5, j_tot=2, f_tot=2.5, species="Yb173"),
    AngularKetFJ(f_c=2, l_r=1, j_r=1.5, f_tot=2.5, species="Yb173"),
    AngularKetFJ(f_c=2, l_r=1, j_r=1.5, f_tot=2.5, species="Yb173"),
]


@pytest.mark.parametrize("ket", TEST_KETS)
def test_exp_q_different_coupling_schemes(ket: AngularKetBase) -> None:
    all_qns: tuple[AngularMomentumQuantumNumbers, ...] = get_args(AngularMomentumQuantumNumbers)
    for q in all_qns:
        exp_q = ket.to_state("LS").calc_exp_qn(q)
        assert np.isclose(exp_q, ket.to_state("JJ").calc_exp_qn(q))
        assert np.isclose(exp_q, ket.to_state("FJ").calc_exp_qn(q))

        std_q = ket.to_state("LS").calc_std_qn(q)
        assert np.isclose(std_q, ket.to_state("JJ").calc_std_qn(q))
        assert np.isclose(std_q, ket.to_state("FJ").calc_std_qn(q))


@pytest.mark.parametrize(("ket1", "ket2"), TEST_KET_PAIRS)
def test_overlap_different_coupling_schemes(ket1: AngularKetBase, ket2: AngularKetBase) -> None:
    ov = ket1.calc_reduced_overlap(ket2)

    coupling_schemes: list[CouplingScheme] = ["LS", "JJ", "FJ"]
    for scheme in coupling_schemes:
        assert np.isclose(ov, ket1.to_state().calc_reduced_overlap(ket2.to_state(scheme)))
        assert np.isclose(ov, ket1.to_state(scheme).calc_reduced_overlap(ket2))
        assert np.isclose(1, ket1.to_state(scheme).calc_reduced_overlap(ket1))
        assert np.isclose(1, ket2.to_state(scheme).calc_reduced_overlap(ket2))


@pytest.mark.parametrize("ket", TEST_KETS)
def test_reduced_identity(ket: AngularKetBase) -> None:
    reduced_identity = np.sqrt(2 * ket.f_tot + 1)

    op: AngularMomentumQuantumNumbers
    coupling_schemes: list[CouplingScheme] = ["LS", "JJ", "FJ"]
    for scheme in coupling_schemes:
        state = ket.to_state(scheme)
        for op in state.kets[0].quantum_number_names:
            assert np.isclose(reduced_identity, state.calc_reduced_matrix_element(state, "identity_" + op, kappa=0))  # type: ignore [arg-type]


@pytest.mark.parametrize(("ket1", "ket2"), TEST_KET_PAIRS)
def test_matrix_elements_in_different_coupling_schemes(ket1: AngularKetBase, ket2: AngularKetBase) -> None:
    example_list: list[tuple[AngularOperatorType, int]] = [
        ("SPHERICAL", 0),
        ("SPHERICAL", 1),
        ("SPHERICAL", 2),
        ("SPHERICAL", 3),
        ("s_tot", 1),
        ("l_r", 1),
        ("i_c", 1),
        ("f_tot", 1),
        ("j_tot", 1),
    ]
    coupling_schemes: list[CouplingScheme] = ["LS", "JJ", "FJ"]

    for scheme in coupling_schemes:
        for operator, kappa in example_list:
            msg = f"{operator=}, {kappa=}, {ket1=}, {ket2=}, {scheme=}"
            val = ket1.calc_reduced_matrix_element(ket2, operator, kappa)

            assert np.isclose(
                val, ket1.to_state().calc_reduced_matrix_element(ket2.to_state(scheme), operator, kappa)
            ), msg
            assert np.isclose(val, ket1.to_state(scheme).calc_reduced_matrix_element(ket2, operator, kappa)), msg
