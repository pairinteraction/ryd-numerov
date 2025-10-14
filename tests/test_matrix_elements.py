import numpy as np
import pytest
from ryd_numerov.rydberg import RydbergState
from ryd_numerov.units import BaseUnits, ureg


@pytest.mark.parametrize("l", [0, 1, 20])
def test_magnetic(l: int) -> None:
    """Test magnetic units."""
    g_s = 2.002319304363
    g_l = 1

    ket = RydbergState("Rb", n=max(l + 1, 10), l=l, j_tot=l + 0.5, m=l + 0.5)

    # Check that for m = j_tot = l + s_tot the magnetic matrix element is - mu_B * (g_l * l + g_s * s_tot)
    mu = ket.calc_matrix_element(ket, "MAGNETIC", 0, 1, 0)
    mu = mu.to("bohr_magneton")
    assert np.isclose(mu.magnitude, -(g_l * l + g_s * 0.5)), f"{mu.magnitude} != {-(g_l * l + g_s * 0.5)}"

    # Check dimensionality
    magnetic_field = ureg.Quantity(1, "T")
    zeeman_energy = -mu * magnetic_field
    assert zeeman_energy.dimensionality == BaseUnits["ENERGY"].dimensionality, (
        f"{zeeman_energy.dimensionality} != {BaseUnits['ENERGY'].dimensionality}"
    )
