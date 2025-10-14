import numpy as np
import pytest
from ryd_numerov.radial import calc_radial_matrix_element
from ryd_numerov.rydberg import RydbergState


@pytest.mark.parametrize(
    ("species", "n", "dn", "dl", "dj"),
    [
        ("Rb", 100, 3, 1, 0),
        ("Rb", 60, 2, 0, 0),
        ("Rb", 81, 2, 2, 2),
        ("Rb", 130, 5, 1, 0),
        ("Rb", 111, 5, 2, 1),
        ("Cs", 60, 2, 0, 0),
        ("K", 81, 2, 2, 2),
    ],
)
def test_circular_matrix_element(species: str, n: int, dn: int, dl: int, dj: int) -> None:
    """Test radial matrix elements of ((almost) circular states, i.e. with large l (l = n-1 for circular states).

     Circular matrix elements should be very close to the perfect hydrogen case, so we can check if the matrix elements
    are reasonable by comparing them to the hydrogen case.
    """
    l, j_tot = n - 1, n - 0.5

    matrix_element = {}
    for _species in [species, "H_textbook"]:
        state_i = RydbergState(_species, n=n, l=l, j_tot=j_tot)  # circular state
        state_f = RydbergState(_species, n=n + dn, l=l + dl, j_tot=j_tot + dj)  # almost circular state
        matrix_element[_species] = calc_radial_matrix_element(state_i, state_f, 1)

    assert np.isclose(matrix_element[species], matrix_element["H_textbook"], rtol=1e-4)


@pytest.mark.parametrize(
    ("species", "n", "l", "j_tot"),
    [
        # for hydrogen the expectation value of r is exact for all states
        ("H", 1, 0, 0.5),
        ("H", 2, 0, 0.5),
        ("H", 2, 1, 0.5),
        ("H", 2, 1, 1.5),
        ("H", 60, 30, 29.5),
        # for other species it is only approximate for circular states
        ("Rb", 100, 99, 99.5),
        ("Rb", 88, 87, 86.5),
    ],
)
def test_circular_expectation_value(species: str, n: int, l: int, j_tot: float) -> None:
    """For circular states, the expectation value of r should be the same as for the hydrogen atom.

    For hydrogen the expectation values of r and r^2 are given by

    .. math::
        <r>_{nl} = 1/2 (3 n^2 - l(l+1))
        <r^2>_{nl} = n^2/2 (5 n^2 - 3 l(l+1) + 1)
    """
    state = RydbergState(species, n=n, l=l, j_tot=j_tot)
    state.create_wavefunction()

    exp_value_numerov = {i: calc_radial_matrix_element(state, state, i) for i in range(3)}
    exp_value_analytic = {
        0: 1,
        1: 0.5 * (3 * n**2 - l * (l + 1)),
        2: n**2 / 2 * (5 * n**2 - 3 * l * (l + 1) + 1),
    }

    for i in range(3):
        assert np.isclose(exp_value_numerov[i], exp_value_analytic[i], rtol=1e-2), (
            f"Expectation value of r^{i} is not correct."
        )
