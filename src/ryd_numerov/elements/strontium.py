from pathlib import Path
from typing import ClassVar, Optional

from ryd_numerov.elements.base_element import BaseElement
from ryd_numerov.units import electron_mass, rydberg_constant


class _StrontiumAbstract(BaseElement):
    Z = 38
    number_valence_electrons = 2
    ground_state_shell = (5, 0)
    _additional_allowed_shells: ClassVar = [(4, 2), (4, 3)]

    _core_electron_configuration = "5s"
    _nist_energy_levels_file = Path(__file__).parent / "nist_energy_levels" / "strontium.txt"

    # https://webbook.nist.gov/cgi/inchi?ID=C7440246&Mask=20
    _ionization_energy: tuple[float, Optional[float], str] = (5.694_84, 0.000_02, "eV")

    potential_type_default = "model_potential_fei_2009"

    # Phys. Rev. A 89, 023426 (2014)
    alpha_c_marinescu_1993 = 7.5
    r_c_dict_marinescu_1993: ClassVar = {0: 1.59, 1: 1.58, 2: 1.57, 3: 1.56}
    model_potential_parameter_marinescu_1993: ClassVar = {
        0: (3.36124, 1.3337, 0, 5.94337),
        1: (3.28205, 1.24035, 0, 3.78861),
        2: (2.155, 1.4545, 0, 4.5111),
        3: (2.1547, 1.14099, 0, 2.1987),
    }

    # https://iopscience.iop.org/article/10.1088/1674-1056/18/10/025
    model_potential_parameter_fei_2009 = (0.9959, 16.9567, 0.2648, 0.1439)


class Strontium88(_StrontiumAbstract):
    species = "Sr88"

    # https://physics.nist.gov/PhysRefData/Handbook/Tables/strontiumtable1.htm
    _isotope_mass = 87.905619  # u
    _corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass),
        None,
        str(rydberg_constant.u),
    )

    # -- [1] Phys. Rev. A 108, 022815 (2023)
    # -- [2] http://dx.doi.org/10.17169/refubium-34581
    # -- [3] Comput. Phys. Commun. 45, 107814 (2021)
    _quantum_defects: ClassVar = {
        # singlet
        (0, 0.0, 0): (3.2688559, -0.0879, -3.36, 0.0, 0.0),  # [1]
        (1, 1.0, 0): (2.7314851, -5.1501, -140.0, 0.0, 0.0),  # [1]
        (2, 2.0, 0): (2.3821857, -40.5009, -878.6, 0.0, 0.0),  # [1]
        (3, 3.0, 0): (0.0873868, -1.5446, 7.56, 0.0, 0.0),  # [1]
        (4, 4.0, 0): (0.038, 0.0, 0.0, 0.0, 0.0),  # [2]
        (5, 5.0, 0): (0.0134759, 0.0, 0.0, 0.0, 0.0),  # [2]
        # triplet
        (0, 1.0, 1): (3.370773, 0.420, -0.4, 0.0, 0.0),  # [3]
        (1, 0.0, 1): (2.8867, 0.43, -1.8, 0.0, 0.0),  # [3]
        (1, 1.0, 1): (2.8826, 0.39, -1.1, 0.0, 0.0),  # [3]
        (1, 2.0, 1): (2.882, -2.5, 100, 0.0, 0.0),  # [3]
        (2, 1.0, 1): (2.67524, -13.23, -4420, 0.0, 0.0),  # [3]
        (2, 2.0, 1): (2.66149, -16.9, -6630, 0.0, 0.0),  # [3]
        (2, 3.0, 1): (2.655, -65, -13577, 0.0, 0.0),  # [3]
        (3, 2.0, 1): (0.120, -2.2, 100, 0.0, 0.0),  # [3]
        (3, 3.0, 1): (0.119, -2.0, 100, 0.0, 0.0),  # [3]
        (3, 4.0, 1): (0.120, -2.4, 120, 0.0, 0.0),  # [3]
    }


class Strontium87(_StrontiumAbstract):
    species = "Sr87"

    # https://physics.nist.gov/PhysRefData/Handbook/Tables/strontiumtable1.htm
    _isotope_mass_u = 86.908884  # u
    _corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass_u),
        None,
        str(rydberg_constant.u),
    )
