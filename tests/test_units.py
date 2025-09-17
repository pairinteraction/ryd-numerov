import numpy as np

from ryd_numerov.units import rydberg_constant, ureg


def test_constants() -> None:
    assert np.isclose(rydberg_constant.to("1/cm", "spectroscopy").magnitude, 109737.31568157, rtol=1e-10, atol=1e-10)
    assert np.isclose(
        ureg.Quantity(1, "fine_structure_constant").to_base_units().magnitude,
        0.0072973525643394025,
        rtol=1e-10,
        atol=1e-10,
    )
