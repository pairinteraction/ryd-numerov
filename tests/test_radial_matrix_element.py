import numpy as np
import pytest
from ryd_numerov.radial import RadialState
from ryd_numerov.rydberg_state import RydbergStateAlkali
from ryd_numerov.species import BaseElement


@pytest.mark.parametrize(
    ("species", "n", "dn", "dl"),
    [
        ("Rb", 100, 3, 1),
        ("Rb", 60, 2, 0),
        ("Rb", 81, 2, 2),
        ("Rb", 130, 5, 1),
        ("Rb", 111, 5, 2),
        ("Cs", 60, 2, 0),
        ("K", 81, 2, 2),
    ],
)
def test_circular_matrix_element(species: str, n: int, dn: int, dl: int) -> None:
    """Test radial matrix elements of ((almost) circular states, i.e. with large l (l = n-1 for circular states).

     Circular matrix elements should be very close to the perfect hydrogen case, so we can check if the matrix elements
    are reasonable by comparing them to the hydrogen case.
    """
    l1 = n - 1  # circular state
    l2 = l1 + dl  # almost circular state

    matrix_element = {}
    for _species in [species, "H_textbook"]:
        state_i = RydbergStateAlkali(_species, n=n, l=l1, j=l1 + 0.5)
        state_f = RydbergStateAlkali(_species, n=n + dn, l=l2, j=l2 + 0.5)
        matrix_element[_species] = state_i.radial.calc_matrix_element(state_f.radial, 1, unit="bohr")

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
    element = BaseElement.from_species(species)
    energy_au = element.calc_energy(n, l, j_tot, unit="hartree")
    nu = element.calc_nu_from_energy(energy_au)

    state = RadialState(species, nu=nu, l_r=l)
    state.set_n_for_sanity_check(n)
    state.create_wavefunction()

    exp_value_numerov = {i: state.calc_matrix_element(state, i, unit=f"bohr^{i}" if i > 0 else "") for i in range(3)}
    exp_value_analytic = {
        0: 1,
        1: 0.5 * (3 * n**2 - l * (l + 1)),
        2: n**2 / 2 * (5 * n**2 - 3 * l * (l + 1) + 1),
    }

    for i in range(3):
        assert np.isclose(exp_value_numerov[i], exp_value_analytic[i], rtol=1e-2), (
            f"Expectation value of r^{i} is not correct."
        )
