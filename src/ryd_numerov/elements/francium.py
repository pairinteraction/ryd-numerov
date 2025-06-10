from typing import ClassVar

from ryd_numerov.elements.element import Element


class Francium(Element):
    species = "Fr"
    s = 1 / 2
    ground_state_shell = (7, 0)

    # https://webbook.nist.gov/cgi/inchi?ID=C7440735&Mask=20
    _ionization_energy = (4.071_2, 0.000_04, "eV")

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
