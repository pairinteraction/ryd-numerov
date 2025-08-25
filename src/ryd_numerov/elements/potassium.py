from pathlib import Path
from typing import ClassVar

from ryd_numerov.elements.base_element import BaseElement


class Potassium(BaseElement):
    species = "K"
    Z = 19
    number_valence_electrons = 1
    ground_state_shell = (4, 0)
    _additional_allowed_shells: ClassVar = [(3, 2)]
    _core_electron_configuration = "3p6"
    _nist_energy_levels_file = Path(__file__).parent / "nist_energy_levels" / "potassium.txt"

    # https://webbook.nist.gov/cgi/inchi?ID=C7440097&Mask=20
    _ionization_energy = (4.340_66, 0.000_01, "eV")

    potential_type_default = "model_potential_marinescu_1993"

    # -- [1] Phys. Scr. 27, 300 (1983)
    # -- [2] Opt. Commun. 39, 370 (1981)
    # -- [3] Ark. Fys., 10 p.583 (1956)
    _quantum_defects: ClassVar = {
        (0, 0.5, 1 / 2): (2.180197, 0.136, 0.0759, 0.117, -0.206),  # [1,2]
        (1, 0.5, 1 / 2): (1.713892, 0.2332, 0.16137, 0.5345, -0.234),  # [1]
        (1, 1.5, 1 / 2): (1.710848, 0.2354, 0.11551, 1.105, -2.0356),  # [1]
        (2, 1.5, 1 / 2): (0.27697, -1.0249, -0.709174, 11.839, -26.689),  # [1,2]
        (2, 2.5, 1 / 2): (0.277158, -1.0256, -0.59201, 10.0053, -19.0244),  # [1,2]
        (3, 2.5, 1 / 2): (0.010098, -0.100224, 1.56334, -12.6851, 0),  # [1,3]
        (3, 3.5, 1 / 2): (0.010098, -0.100224, 1.56334, -12.6851, 0),  # [1,3]
    }

    _corrected_rydberg_constant = (109735.774, None, "1/cm")

    # Phys. Rev. A 49, 982 (1994)
    alpha_c_marinescu_1993 = 5.3310
    r_c_dict_marinescu_1993: ClassVar = {0: 0.83167545, 1: 0.85235381, 2: 0.83216907, 3: 6.50294371}
    model_potential_parameter_marinescu_1993: ClassVar = {
        0: (3.56079437, 1.83909642, -1.74701102, -1.03237313),
        1: (3.65670429, 1.67520788, -2.07416615, -0.89030421),
        2: (4.12713694, 1.79837462, -1.69935174, -0.98913582),
        3: (1.42310446, 1.27861156, 4.77441476, -0.94829262),
    }

    # https://iopscience.iop.org/article/10.1088/1674-1056/18/10/025
    model_potential_parameter_fei_2009 = (0.9172, 4.1728, 0.6845, 0.2280)
