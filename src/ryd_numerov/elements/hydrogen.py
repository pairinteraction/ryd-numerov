from ryd_numerov.elements.element import Element
from ryd_numerov.units import ureg

RydbergConstant = ureg.Quantity(1, "rydberg_constant").to("eV", "spectroscopy")


class Hydrogen(Element):
    species = "H"
    Z = 1
    s = 1 / 2
    ground_state_shell = (1, 0)

    # https://webbook.nist.gov/cgi/inchi?ID=C1333740&Mask=20
    _ionization_energy = (15.425_93, 0.000_05, "eV")

    _corrected_rydberg_constant = (109677.58340280356, None, "1/cm")


class HydrogenTextBook(Element):
    """Hydrogen from QM textbook with infinite nucleus mass and no spin orbit coupling."""

    species = "H_textbook"
    s = 1 / 2
    ground_state_shell = (1, 0)

    _ionization_energy = (RydbergConstant.magnitude, 0, str(RydbergConstant.units))

    _corrected_rydberg_constant = (109737.31568160003, None, "1/cm")
