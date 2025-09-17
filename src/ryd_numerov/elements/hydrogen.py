from typing import ClassVar

from ryd_numerov.elements.base_element import BaseElement
from ryd_numerov.units import rydberg_constant


class Hydrogen(BaseElement):
    species = "H"
    Z = 1
    number_valence_electrons = 1
    ground_state_shell = (1, 0)

    # https://webbook.nist.gov/cgi/inchi?ID=C1333740&Mask=20
    _ionization_energy = (15.425_93, 0.000_05, "eV")

    potential_type_default = "coulomb"

    _quantum_defects: ClassVar = {}

    _corrected_rydberg_constant = (109677.58340280356, None, "1/cm")


class HydrogenTextBook(BaseElement):
    """Hydrogen from QM textbook with infinite nucleus mass and no spin orbit coupling."""

    species = "H_textbook"
    number_valence_electrons = 1
    ground_state_shell = (1, 0)

    _ionization_energy = (rydberg_constant.m, 0, str(rydberg_constant.u))

    potential_type_default = "coulomb"

    _quantum_defects: ClassVar = {}

    _corrected_rydberg_constant = (109737.31568160003, None, "1/cm")
