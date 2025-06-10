from pathlib import Path
from typing import ClassVar, Optional

from ryd_numerov.elements.base_element import BaseElement


class _StrontiumAbstract(BaseElement):
    Z = 38

    _core_electron_configuration = "5s"
    _nist_energy_levels_file = Path(__file__).parent / "nist_energy_levels" / "strontium.txt"

    # https://webbook.nist.gov/cgi/inchi?ID=C7440246&Mask=20
    _ionization_energy: tuple[float, Optional[float], str] = (5.694_84, 0.000_02, "eV")

    additional_potentials_default: ClassVar = ["spin_orbit", "core_corrections", "core_polarization"]

    # http://dollywood.itp.tuwien.ac.at/~shuhei/pub_Rydberg/PhysRevA.89.023426.pdf
    alpha_c = 7.5
    # http://dollywood.itp.tuwien.ac.at/~shuhei/pub_Rydberg/PhysRevA.89.023426.pdf
    _r_c_dict: ClassVar = {0: 1.59, 1: 1.58, 2: 1.57, 3: 1.56}
    # http://dollywood.itp.tuwien.ac.at/~shuhei/pub_Rydberg/PhysRevA.89.023426.pdf
    _parametric_model_potential_parameters: ClassVar = {
        0: (3.36124, 1.3337, 0, 5.94337),
        1: (3.28205, 1.24035, 0, 3.78861),
        2: (2.155, 1.4545, 0, 4.5111),
        3: (2.1547, 1.14099, 0, 2.1987),
    }


class Strontium88Singlet(_StrontiumAbstract):
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


class Strontium88Triplet(_StrontiumAbstract):
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
