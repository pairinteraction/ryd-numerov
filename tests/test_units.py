from ryd_numerov.units import ureg


def test_constants() -> None:
    assert ureg.Quantity(1, "rydberg_constant").to("1/cm").magnitude == 109737.31568157
    assert ureg.Quantity(1, "fine_structure_constant").to_base_units().magnitude == 0.0072973525643394025
