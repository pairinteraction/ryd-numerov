import numpy as np
import pytest
from sympy.abc import r as sympy_r
from sympy.physics import hydrogen
from sympy.utilities.lambdify import lambdify

from numerov.rydberg import RydbergState


@pytest.mark.parametrize(
    "species, n, l, run_backward",
    [
        ("H", 1, 0, True),  # Ground state
        ("H", 2, 0, True),  # First excited s state
        ("H", 2, 1, True),  # First p state
        ("H", 3, 1, True),  # n=3, l=1 state
        ("H", 3, 2, True),  # n=3, l=2 state
        ("He+", 2, 0, True),  # He+ ground state
        ("H", 4, 3, False),
    ],
)
def test_hydrogen_wavefunctions(species: str, n: int, l: int, run_backward: bool):
    """Test that numerov integration matches sympy's analytical hydrogen wavefunctions."""
    # Setup atom
    atom = RydbergState(species, n, l, j=l + 0.5, run_backward=run_backward)

    # Run the numerov integration
    x_list, u_list = atom.integrate()

    # Get analytical solution from sympy
    Z = {"H": 1, "He+": 2}[species]
    R_nl = lambdify(sympy_r, hydrogen.R_nl(n, l, sympy_r, Z))
    u_list_sympy = R_nl(x_list) * x_list

    # Compare numerical and analytical solutions
    np.testing.assert_allclose(u_list, u_list_sympy, rtol=1e-2, atol=1e-2)
