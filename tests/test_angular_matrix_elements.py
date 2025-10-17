from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ryd_numerov.angular import AngularKetFJ, AngularKetJJ, AngularKetLS
from ryd_numerov.units import OperatorType

if TYPE_CHECKING:
    from ryd_numerov.angular import AngularKetBase


@pytest.mark.parametrize(
    ("ket", "q"),
    [
        (AngularKetLS(s_tot=1, l_r=1, j_tot=1, f_tot=1.5, species="Yb173"), "s_tot"),
        (AngularKetLS(s_tot=1, l_r=1, j_tot=1, f_tot=1.5, species="Yb173"), "f_c"),
        (AngularKetJJ(l_r=1, j_r=1.5, j_tot=2, f_tot=2.5, species="Yb173"), "s_tot"),
        (AngularKetFJ(f_c=2, l_r=1, j_r=1.5, f_tot=2.5, species="Yb173"), "s_tot"),
        (AngularKetFJ(f_c=2, l_r=1, j_r=1.5, f_tot=2.5, species="Yb173"), "j_r"),
    ],
)
def test_exp_q_different_coupling_schemes(ket: AngularKetBase, q: str) -> None:
    exp_q = ket.to_ls().calc_exp_qn(q)
    assert np.isclose(exp_q, ket.to_jj().calc_exp_qn(q))
    assert np.isclose(exp_q, ket.to_fj().calc_exp_qn(q))

    std_q = ket.to_ls().calc_std_qn(q)
    assert np.isclose(std_q, ket.to_jj().calc_std_qn(q))
    assert np.isclose(std_q, ket.to_fj().calc_std_qn(q))


@pytest.mark.parametrize(
    ("ket1", "ket2"),
    [
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
    ],
)
def test_overlap_different_coupling_schemes(ket1: AngularKetBase, ket2: AngularKetBase) -> None:
    ov = ket1.calc_reduced_overlap(ket2)
    state1 = ket1.to_state()
    state2 = ket2.to_state()
    assert np.isclose(ov, state1.calc_reduced_overlap(ket2.to_ls()))
    assert np.isclose(ov, state1.calc_reduced_overlap(ket2.to_jj()))
    assert np.isclose(ov, state1.calc_reduced_overlap(ket2.to_fj()))
    assert np.isclose(ov, ket1.to_ls().calc_reduced_overlap(state2))
    assert np.isclose(ov, ket1.to_jj().calc_reduced_overlap(state2))
    assert np.isclose(ov, ket1.to_fj().calc_reduced_overlap(state2))


@pytest.mark.parametrize(
    ("ket1", "ket2"),
    [
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
    ],
)
def test_matrix_elements_in_different_coupling_schemes(ket1: AngularKetBase, ket2: AngularKetBase) -> None:
    operator: OperatorType
    for operator, kappa in [
        ("SPHERICAL", 0),
        ("SPHERICAL", 1),
        ("SPHERICAL", 2),
        ("SPHERICAL", 3),
        ("s_tot", 1),
        ("l_r", 1),
        ("i_c", 1),
        ("f_tot", 1),
        ("j_tot", 1),
    ]:
        val = ket1.calc_reduced_matrix_element(ket2, operator, kappa)
        state1 = ket1.to_ls()
        state2 = ket2.to_ls()
        assert np.isclose(val, state1.calc_reduced_matrix_element(ket2.to_ls(), operator, kappa))
        assert np.isclose(val, state1.calc_reduced_matrix_element(ket2.to_jj(), operator, kappa))
        assert np.isclose(val, state1.calc_reduced_matrix_element(ket2.to_fj(), operator, kappa))
        assert np.isclose(val, ket1.to_ls().calc_reduced_matrix_element(state2, operator, kappa))
        assert np.isclose(val, ket1.to_jj().calc_reduced_matrix_element(state2, operator, kappa))
        assert np.isclose(val, ket1.to_fj().calc_reduced_matrix_element(state2, operator, kappa))
