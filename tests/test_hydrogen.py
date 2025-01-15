import numpy as np
import pytest
from sympy.abc import r as sympy_r
from sympy.physics import hydrogen
from sympy.utilities.lambdify import lambdify

from numerov.rydberg import RydbergState


@pytest.mark.parametrize(
    "species, n, l, run_backward",
    [
        ("H", 1, 0, True),
        ("H", 2, 0, True),
        ("H", 2, 1, True),
        ("H", 2, 1, False),
        ("H", 3, 0, True),
        ("H", 3, 2, True),
        ("H", 3, 2, False),
        ("H", 30, 0, True),
        ("H", 30, 1, True),
        ("H", 30, 28, True),
        ("H", 30, 29, True),
        ("He+", 2, 0, True),
    ],
)
def test_hydrogen_wavefunctions(species: str, n: int, l: int, run_backward: bool):
    """Test that numerov integration matches sympy's analytical hydrogen wavefunctions."""
    # Setup atom
    atom = RydbergState(species, n, l, j=l + 0.5, run_backward=run_backward)

    # Run the numerov integration
    atom.integrate()

    # Get analytical solution from sympy
    Z = {"H": 1, "He+": 2}[species]
    R_nl_lambda = lambdify(sympy_r, hydrogen.R_nl(n, l, sympy_r, Z))
    R_nl = R_nl_lambda(atom.x_list)

    # Compare numerical and analytical solutions
    np.testing.assert_allclose(atom.R_list, R_nl, rtol=1e-2, atol=1e-2)
