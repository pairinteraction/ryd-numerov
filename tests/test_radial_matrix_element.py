import numpy as np
import pytest

from numerov.matrix_elements import radial_matrix_element
from numerov.rydberg import RydbergState


@pytest.mark.parametrize(
    "n, dn, dl, dj",
    [
        (100, 3, 1, 0),
        (60, 2, 0, 0),
        (81, 2, 2, 2),
        (130, 5, 1, 0),
        (111, 5, 2, 1),
    ],
)
def test_circular_matrix_elements(n: int, dn: int, dl: int, dj: int) -> None:
    """Test radial matrix elements of ((almost) circular states, i.e. with large l (l = n-1 for circular states).

     Circular matrix elements should be very close to the perfect hydrogen case, so we can check if the matrix elements
    are reasonable by comparing them to the hydrogen case.
    """
    l, j = n - 1, n - 0.5

    matrix_element = {}
    for species in ["Rb", "H"]:
        state_i = RydbergState("Rb", n, l, j)  # circular state
        state_i.create_model(add_spin_orbit=species != "H")
        state_i.integrate()

        state_f = RydbergState("Rb", n + dn, l + dl, j + dj)  # almost circular state
        state_f.create_model(add_spin_orbit=species != "H")
        state_f.integrate()

        matrix_element[species] = radial_matrix_element(state_i, state_f, 1)

    assert np.isclose(matrix_element["Rb"], matrix_element["H"], rtol=1e-4)


@pytest.mark.parametrize(
    "species, n, l, j",
    [
        ("H", 60, 30, 29.5),  # for hydrogen the expectation value of r is exact for all states
        ("Rb", 100, 99, 99.5),
        ("Rb", 88, 87, 86.5),
    ],
)
def test_circular_expectation_value(species: str, n: int, l: int, j: float) -> None:
    """For circular states, the expectation value of r should be the same as for the hydrogen atom.

    For hydrogen the expectation values of r and r^2 are given by

    .. math::
        <r>_{nl} = 1/2 (3 n^2 - l(l+1))
        <r^2>_{nl} = n^2/2 (5 n^2 - 3 l(l+1) + 1)
    """
    state = RydbergState(species, n, l, j)
    state.create_model(add_spin_orbit=species != "H")
    state.integrate()

    exp_value_numerov = {i: radial_matrix_element(state, state, i) for i in range(3)}
    exp_value_analytic = {
        0: 1,
        1: 0.5 * (3 * n**2 - l * (l + 1)),
        2: n**2 / 2 * (5 * n**2 - 3 * l * (l + 1) + 1),
    }

    for i in range(3):
        assert np.isclose(exp_value_numerov[i], exp_value_analytic[i], rtol=1e-2), (
            f"Expectation value of r^{i} is not correct."
        )
