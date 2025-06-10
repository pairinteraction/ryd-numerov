from typing import ClassVar

from ryd_numerov.elements.element import Element


class Cesium(Element):
    species = "Cs"
    s = 1 / 2
    ground_state_shell = (6, 0)

    # https://webbook.nist.gov/cgi/inchi?ID=C7440462&Mask=20
    _ionization_energy = (3.893_90, 0.000_002, "eV")

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
