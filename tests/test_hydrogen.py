import numpy as np
import pytest
from sympy.abc import r as sympy_r
from sympy.physics import hydrogen as sympy_hydrogen
from sympy.utilities.lambdify import lambdify

from ryd_numerov.rydberg import RydbergState


@pytest.mark.parametrize(
    ("species", "n", "l", "run_backward"),
    [
        ("H_textbook", 1, 0, True),
        ("H_textbook", 2, 0, True),
        ("H_textbook", 2, 1, True),
        ("H_textbook", 2, 1, False),
        ("H_textbook", 3, 0, True),
        ("H_textbook", 3, 2, True),
        ("H_textbook", 3, 2, False),
        ("H_textbook", 30, 0, True),
        ("H_textbook", 30, 1, True),
        ("H_textbook", 30, 2, True),
        ("H_textbook", 30, 28, True),
        ("H_textbook", 30, 29, True),
        ("H_textbook", 130, 128, True),
        ("H_textbook", 130, 129, True),
    ],
)
def test_hydrogen_wavefunctions(species: str, n: int, l: int, run_backward: bool) -> None:
    """Test that numerov integration matches sympy's analytical hydrogen wavefunctions."""
    # Setup atom
    atom = RydbergState(species, n=n, l=l, j=l + 0.5)

    # Run the numerov integration
    atom.create_wavefunction(run_backward=run_backward)

    # Get analytical solution from sympy
    if n <= 35:
        r_nl_lambda = lambdify(sympy_r, sympy_hydrogen.R_nl(n, l, sympy_r, Z=1))
        r_nl = r_nl_lambda(atom.grid.x_list)
    else:  # some weird sympy bug if trying to use lambdify R_nl for n > 35
        r_nl = np.zeros_like(atom.grid.x_list)
        for i, x in enumerate(atom.grid.x_list):
            r_nl[i] = sympy_hydrogen.R_nl(n, l, x, Z=1)

    # Compare numerical and analytical solutions
    np.testing.assert_allclose(atom.wavefunction.r_list, r_nl, rtol=1e-2, atol=1e-2)
