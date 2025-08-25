from pathlib import Path
from typing import ClassVar

from ryd_numerov.elements.base_element import BaseElement


class _YtterbiumAbstract(BaseElement):
    Z = 70
    number_valence_electrons = 2
    ground_state_shell = (6, 0)
    _additional_allowed_shells: ClassVar = [(5, 2), (5, 3), (5, 4)]

    _core_electron_configuration = "4f14.6s"
    _nist_energy_levels_file = Path(__file__).parent / "nist_energy_levels" / "ytterbium.txt"

    # https://webbook.nist.gov/cgi/inchi?ID=C7440644&Mask=20
    _ionization_energy = (6.25416, None, "eV")

    potential_type_default = "model_potential_fei_2009"

    # https://iopscience.iop.org/article/10.1088/1674-1056/18/10/025
    model_potential_parameter_fei_2009 = (0.8704, 22.0040, 0.1513, 0.3306)


class Ytterbium174(_YtterbiumAbstract):
    species = "Yb174"

    # https://physics.nist.gov/PhysRefData/Handbook/Tables/ytterbiumtable1.htm
    # _isotope_mass = 173.938859  # u  # noqa: ERA001
    # _corrected_rydberg_constant = Ry_inf / (1 + electron_mass / _isotope_mass)  # noqa: ERA001
    _corrected_rydberg_constant = (109736.96958583764, None, "1/cm")
