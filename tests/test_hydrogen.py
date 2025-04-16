import numpy as np
import pytest
from sympy.abc import r as sympy_r
from sympy.physics import hydrogen as sympy_hydrogen
from sympy.utilities.lambdify import lambdify

from ryd_numerov.rydberg import RydbergState


@pytest.mark.parametrize(
    ("species", "n", "l", "run_backward"),
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
        ("H", 30, 2, True),
        ("H", 30, 28, True),
        ("H", 30, 29, True),
        ("H", 130, 128, True),
        ("H", 130, 129, True),
    ],
)
def test_hydrogen_wavefunctions(species: str, n: int, l: int, run_backward: bool) -> None:
    """Test that numerov integration matches sympy's analytical hydrogen wavefunctions."""
    # Setup atom
    atom = RydbergState(species, n, l, j=l + 0.5)
    atom.create_model(add_spin_orbit=False)

    # Run the numerov integration
    atom.integrate_wavefunction(run_backward=run_backward)

    # Get analytical solution from sympy
    if n <= 35:
        r_nl_lambda = lambdify(sympy_r, sympy_hydrogen.R_nl(n, l, sympy_r, Z=1))
        r_nl = r_nl_lambda(atom.grid.xlist)
    else:  # some weird sympy bug if trying to use lambdify R_nl for n > 35
        r_nl = np.zeros_like(atom.grid.xlist)
        for i, x in enumerate(atom.grid.xlist):
            r_nl[i] = sympy_hydrogen.R_nl(n, l, x, Z=1)

    # Compare numerical and analytical solutions
    np.testing.assert_allclose(atom.wavefunction.r_list, r_nl, rtol=1e-2, atol=1e-2)
