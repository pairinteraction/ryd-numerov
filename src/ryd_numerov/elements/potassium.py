from typing import ClassVar

from ryd_numerov.elements.element import Element


class Potassium(Element):
    species = "K"
    s = 1 / 2
    ground_state_shell = (4, 0)

    # https://webbook.nist.gov/cgi/inchi?ID=C7440097&Mask=20
    _ionization_energy = (4.340_66, 0.000_01, "eV")

    # -- [1] Phys. Scr. 27, 300 (1983)
    # -- [2] Opt. Commun. 39, 370 (1981)
    # -- [3] Ark. Fys., 10 p.583 (1956)
    _quantum_defects: ClassVar = {
        (0, 0.5): (2.180197, 0.136, 0.0759, 0.117, -0.206),  # [1,2]
        (1, 0.5): (1.713892, 0.2332, 0.16137, 0.5345, -0.234),  # [1]
        (1, 1.5): (1.710848, 0.2354, 0.11551, 1.105, -2.0356),  # [1]
        (2, 1.5): (0.27697, -1.0249, -0.709174, 11.839, -26.689),  # [1,2]
        (2, 2.5): (0.277158, -1.0256, -0.59201, 10.0053, -19.0244),  # [1,2]
        (3, 2.5): (0.010098, -0.100224, 1.56334, -12.6851, 0),  # [1,3]
        (3, 3.5): (0.010098, -0.100224, 1.56334, -12.6851, 0),  # [1,3]
    }

    _corrected_rydberg_constant = (109735.774, None, "1/cm")
