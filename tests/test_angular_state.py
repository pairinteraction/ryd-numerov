import numpy as np
from ryd_numerov.angular import AngularStateFJ, AngularStateJJ, AngularStateLS


def test_fj_to_ls() -> None:
    fj = AngularStateFJ(f_c=2, l_r=0, f_tot=2.5, species="Yb173")
    ls_1 = AngularStateLS(s_tot=0, l_r=0, j_tot=0, species="Yb173")
    ls_2 = AngularStateLS(s_tot=1.0, l_r=0, j_tot=1.0, f_tot=2.5, species="Yb173")
    assert np.isclose(fj.calc_reduced_overlap(ls_1), -np.sqrt(5) / 2 / np.sqrt(3))
    assert np.isclose(fj.calc_reduced_overlap(ls_2), np.sqrt(7) / 2 / np.sqrt(3))

    fj_as_ls = fj.to_ls()
    assert len(fj_as_ls.states) == 2
    assert ls_1 in fj_as_ls.states
    assert ls_2 in fj_as_ls.states
    assert fj_as_ls.coefficients[fj_as_ls.states.index(ls_1)] == fj.calc_reduced_overlap(ls_1)
    assert fj_as_ls.coefficients[fj_as_ls.states.index(ls_2)] == fj.calc_reduced_overlap(ls_2)


def test_jj_to_ls() -> None:
    jj = AngularStateJJ(j_tot=0, l_r=0, f_tot=2.5, species="Yb173")
    ls = AngularStateLS(s_tot=0, l_r=0, j_tot=0, species="Yb173")
    assert np.isclose(jj.calc_reduced_overlap(ls), 1.0)

    jj_as_ls = jj.to_ls()
    assert len(jj_as_ls.states) == 1
    assert ls in jj_as_ls.states
    assert jj_as_ls.coefficients[jj_as_ls.states.index(ls)] == jj.calc_reduced_overlap(ls)
