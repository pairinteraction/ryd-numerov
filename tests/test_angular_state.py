import numpy as np
from ryd_numerov.angular import AngularKetFJ, AngularKetJJ, AngularKetLS


def test_fj_to_ls() -> None:
    fj = AngularKetFJ(f_c=2, l_r=0, f_tot=2.5, species="Yb173")
    ls_1 = AngularKetLS(s_tot=0, l_r=0, j_tot=0, species="Yb173")
    ls_2 = AngularKetLS(s_tot=1.0, l_r=0, j_tot=1.0, f_tot=2.5, species="Yb173")
    assert np.isclose(fj.calc_reduced_overlap(ls_1), -np.sqrt(5) / 2 / np.sqrt(3))
    assert np.isclose(fj.calc_reduced_overlap(ls_2), np.sqrt(7) / 2 / np.sqrt(3))

    fj_as_ls = fj.to_state("LS")
    assert len(fj_as_ls.kets) == 2
    assert ls_1 in fj_as_ls.kets
    assert ls_2 in fj_as_ls.kets
    assert fj_as_ls.coefficients[fj_as_ls.kets.index(ls_1)] == fj.calc_reduced_overlap(ls_1)
    assert fj_as_ls.coefficients[fj_as_ls.kets.index(ls_2)] == fj.calc_reduced_overlap(ls_2)


def test_jj_to_ls() -> None:
    jj = AngularKetJJ(j_tot=0, l_r=0, f_tot=2.5, species="Yb173")
    ls = AngularKetLS(s_tot=0, l_r=0, j_tot=0, species="Yb173")
    assert np.isclose(jj.calc_reduced_overlap(ls), 1.0)

    jj_as_ls = jj.to_state("LS")
    assert len(jj_as_ls.kets) == 1
    assert ls in jj_as_ls.kets
    assert jj_as_ls.coefficients[jj_as_ls.kets.index(ls)] == jj.calc_reduced_overlap(ls)


def test_ls_to_jj() -> None:
    ls = AngularKetLS(s_tot=0, l_r=0, j_tot=0, species="Yb173")
    jj = AngularKetJJ(j_tot=0, l_r=0, f_tot=2.5, species="Yb173")
    assert np.isclose(ls.calc_reduced_overlap(jj), 1.0)

    ls_as_jj = ls.to_state("JJ")
    assert len(ls_as_jj.kets) == 1
    assert jj in ls_as_jj.kets
    assert ls_as_jj.coefficients[ls_as_jj.kets.index(jj)] == ls.calc_reduced_overlap(jj)


def test_ls_to_fj() -> None:
    ls = AngularKetLS(s_tot=0, l_r=0, j_tot=0, species="Yb173")
    fj_1 = AngularKetFJ(f_c=2, l_r=0, f_tot=2.5, species="Yb173")
    fj_2 = AngularKetFJ(f_c=3, l_r=0, f_tot=2.5, species="Yb173")
    assert np.isclose(ls.calc_reduced_overlap(fj_1), -np.sqrt(5) / 2 / np.sqrt(3))
    assert np.isclose(ls.calc_reduced_overlap(fj_2), np.sqrt(7) / 2 / np.sqrt(3))

    ls_as_fj = ls.to_state("FJ")
    assert len(ls_as_fj.kets) == 2
    assert fj_1 in ls_as_fj.kets
    assert fj_2 in ls_as_fj.kets
    assert ls_as_fj.coefficients[ls_as_fj.kets.index(fj_1)] == ls.calc_reduced_overlap(fj_1)
    assert ls_as_fj.coefficients[ls_as_fj.kets.index(fj_2)] == ls.calc_reduced_overlap(fj_2)
