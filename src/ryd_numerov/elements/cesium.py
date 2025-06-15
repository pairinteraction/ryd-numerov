from pathlib import Path
from typing import ClassVar

from ryd_numerov.elements.base_element import BaseElement


class Cesium(BaseElement):
    species = "Cs"
    Z = 55
    s = 1 / 2
    ground_state_shell = (6, 0)
    _additional_allowed_shells: ClassVar = [(4, 3), (5, 2), (5, 3), (5, 4)]
    _core_electron_configuration = "5p6"
    _nist_energy_levels_file = Path(__file__).parent / "nist_energy_levels" / "cesium.txt"

    # https://webbook.nist.gov/cgi/inchi?ID=C7440462&Mask=20
    _ionization_energy = (3.893_90, 0.000_002, "eV")

    potential_type_default = "model_potential_marinescu_1993"

    # -- [1] Phys. Rev. A 93, 013424 (2016)
    # -- [2] Phys. Rev. A 26, 2733 (1982)
    # -- [3] Phys. Rev. A 35, 4650 (1987)
    _quantum_defects: ClassVar = {
        (0, 0.5): (4.0493532, 0.2391, 0.06, 11, -209),  # [1]
        (1, 0.5): (3.5915871, 0.36273, 0.0, 0.0, 0.0),  # [1]
        (1, 1.5): (3.5590676, 0.37469, 0.0, 0.0, 0.0),  # [1]
        (2, 1.5): (2.475365, 0.5554, 0.0, 0.0, 0.0),  # [2]
        (2, 2.5): (2.4663144, 0.01381, -0.392, -1.9, 0.0),  # [1]
        (3, 2.5): (0.03341424, -0.198674, 0.28953, -0.2601, 0.0),  # [3]
        (3, 3.5): (0.033537, -0.191, 0.0, 0.0, 0.0),  # [2]
        (4, 3.5): (0.00703865, -0.049252, 0.01291, 0.0, 0.0),  # [3]
    }

    _corrected_rydberg_constant = (109736.8627339, None, "1/cm")

    # Phys. Rev. A 49, 982 (1994)
    alpha_c_marinescu_1993 = 15.6440
    r_c_dict_marinescu_1993: ClassVar = {0: 1.92046930, 1: 2.13383095, 2: 0.93007296, 3: 1.99969677}
    model_potential_parameter_marinescu_1993: ClassVar = {
        0: (3.49546309, 1.47533800, -9.72143084, 0.02629242),
        1: (4.69366096, 1.71398344, -24.65624280, -0.09543125),
        2: (4.32466196, 1.61365288, -6.70128850, -0.74095193),
        3: (3.01048361, 1.40000001, -3.20036138, 0.00034538),
    }

    # https://iopscience.iop.org/article/10.1088/1674-1056/18/10/025
    model_potential_parameter_fei_2009 = (0.9447, 14.7149, 0.2944, 0.1934)
