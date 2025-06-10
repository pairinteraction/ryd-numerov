from pathlib import Path
from typing import ClassVar

from ryd_numerov.elements.base_element import BaseElement


class Francium(BaseElement):
    species = "Fr"
    Z = 87
    s = 1 / 2
    ground_state_shell = (7, 0)
    _core_electron_configuration = "6p6"
    _nist_energy_levels_file = Path(__file__).parent / "nist_energy_levels" / "francium.txt"

    # https://webbook.nist.gov/cgi/inchi?ID=C7440735&Mask=20
    _ionization_energy = (4.071_2, 0.000_04, "eV")

    additional_potentials_default: ClassVar = ["spin_orbit", "core_corrections", "core_polarization"]

    # -- [1] Phys. Rev. A 86, 052518 (2012)
    # -- [2] Phys. Rev. A 93, 042506 (2016)
    _quantum_defects: ClassVar = {
        (0, 0.5): (4.277896, 0.2332, 0.0348, 0.0, 0.0),  # [1]
        (1, 0.5): (3.715424, 0.3865, 0.0006, 0.0, 0.0),  # [1]
        (1, 1.5): (3.692354, 0.3780, 0.0006, 0.0, 0.0),  # [1]
        (2, 1.5): (2.654543, 0.0612, 0.0, 0.0, 0.0),  # [2]
        (2, 2.5): (2.639608, 0.0589, 0.0, 0.0, 0.0),  # [2]
        (3, 2.5): (0.033421, -0.188, 0.0, 0.0, 0.0),  # [2]
        (3, 3.5): (0.033421, -0.191, 0.0, 0.0, 0.0),  # [2]
    }

    _corrected_rydberg_constant = (109_736.862_733_9, None, "1/cm")

    # J. Phys. B 43 202001 (2010)
    alpha_c = 20.4
    # Phys. Rev. A 49, 982 (1994)
    _r_c_dict: ClassVar = {  # Using Cs values as approximation
        0: 3.49546309,
        1: 4.69366096,
        2: 4.32466196,
        3: 3.01048361,
    }
    # Phys. Rev. A 49, 982 (1994)
    _parametric_model_potential_parameters: ClassVar = {
        0: (1.47533800, -9.72143084, 0.02629242, 1.92046930),  # Using Cs values as approximation
        1: (1.71398344, -24.65624280, -0.09543125, 2.13383095),  # Using Cs values as approximation
        2: (1.61365288, -6.70128850, -0.74095193, 0.93007296),  # Using Cs values as approximation
        3: (1.40000001, -3.20036138, 0.00034538, 1.99969677),  # Using Cs values as approximation
    }
