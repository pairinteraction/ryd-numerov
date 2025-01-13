import numpy as np
import pytest
from sympy.abc import r as sympy_r
from sympy.physics import hydrogen
from sympy.utilities.lambdify import lambdify

from numerov.integrate import integrate_hydrogen


@pytest.mark.parametrize(
    "n, l, Z, direction",
    [
        (1, 0, 1, "backward"),  # Ground state
        (2, 0, 1, "backward"),  # First excited s state
        (2, 1, 1, "backward"),  # First p state
        (3, 1, 1, "backward"),  # n=3, l=1 state
        (3, 2, 1, "backward"),  # n=3, l=2 state
        (2, 0, 2, "backward"),  # He+ ground state
        (4, 3, 1, "forward"),
    ],
)
def test_hydrogen_wavefunctions(n, l, Z, direction):
    """Test that numerov integration matches sympy's analytical hydrogen wavefunctions."""
    # Set up integration parameters
    energy = -(Z**2) / (n**2)  # Energy in atomic units
    dx = 1e-3
    xmin = dx
    xmax = 80
    # direction = "backward"
    epsilon_u = 1e-10

    # Run numerov integration
    x_list, u_list = integrate_hydrogen(energy, Z, n, l, dx, xmin, xmax, direction, epsilon_u)

    # Get analytical solution from sympy
    R_nl = lambdify(sympy_r, hydrogen.R_nl(n, l, sympy_r, Z))
    u_list_sympy = R_nl(x_list) * x_list

    # Compare numerical and analytical solutions
    np.testing.assert_allclose(u_list, u_list_sympy, rtol=1e-2, atol=1e-2)
