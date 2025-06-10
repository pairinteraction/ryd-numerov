from typing import ClassVar, Optional

from ryd_numerov.elements.element import Element


class Strontium(Element):
    # https://webbook.nist.gov/cgi/inchi?ID=C7440246&Mask=20
    _ionization_energy: tuple[float, Optional[float], str] = (5.694_84, 0.000_02, "eV")


class Strontium88Singlet(Strontium):
    species = "Sr88_singlet"
    s = 0
    ground_state_shell = (5, 0)

    # -- [1] Phys. Rev. A 108, 022815 (2023)
    # -- [2] http://dx.doi.org/10.17169/refubium-34581
    _quantum_defects: ClassVar = {
        (0, 0.0): (3.2688559, -0.0879, -3.36, 0.0, 0.0),  # [1]
        (1, 1.0): (2.7314851, -5.1501, -140.0, 0.0, 0.0),  # [1]
        (2, 2.0): (2.3821857, -40.5009, -878.6, 0.0, 0.0),  # [1]
        (3, 3.0): (0.0873868, -1.5446, 7.56, 0.0, 0.0),  # [1]
        (4, 4.0): (0.038, 0.0, 0.0, 0.0, 0.0),  # [2]
        (5, 5.0): (0.0134759, 0.0, 0.0, 0.0, 0.0),  # [2]
    }

    _corrected_rydberg_constant = (109736.631, None, "1/cm")


class Strontium88Triplet(Strontium):
    species = "Sr88_triplet"
    s = 1
    ground_state_shell = (4, 2)

    # -- [1] Comput. Phys. Commun. 45, 107814 (2021)
    _quantum_defects: ClassVar = {
        (0, 1.0): (3.370773, 0.420, -0.4, 0.0, 0.0),  # [1]
        (1, 2.0): (2.882, -2.5, 100, 0.0, 0.0),  # [1]
        (1, 1.0): (2.8826, 0.39, -1.1, 0.0, 0.0),  # [1]
        (1, 0.0): (2.8867, 0.43, -1.8, 0.0, 0.0),  # [1]
        (2, 3.0): (2.655, -65, -13577, 0.0, 0.0),  # [1]
        (2, 2.0): (2.66149, -16.9, -6630, 0.0, 0.0),  # [1]
        (2, 1.0): (2.67524, -13.23, -4420, 0.0, 0.0),  # [1]
        (3, 4.0): (0.120, -2.4, 120, 0.0, 0.0),  # [1]
        (3, 3.0): (0.119, -2.0, 100, 0.0, 0.0),  # [1]
        (3, 2.0): (0.120, -2.2, 100, 0.0, 0.0),  # [1]
    }

    _corrected_rydberg_constant = (109736.631, None, "1/cm")
