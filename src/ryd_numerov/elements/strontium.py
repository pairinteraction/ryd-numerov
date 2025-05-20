from abc import ABC
from typing import Optional

from ryd_numerov.elements.element import Element


class Strontium(ABC):  # noqa: B024
    # https://webbook.nist.gov/cgi/inchi?ID=C7440246&Mask=20
    _ionization_energy: tuple[float, Optional[float], str] = (5.694_84, 0.000_02, "eV")


class Strontium88Singlet(Strontium, Element):
    species = "Sr88_singlet"
    s = 0
    ground_state_shell = (5, 0)


class Strontium88Triplet(Strontium, Element):
    species = "Sr88_triplet"
    s = 1
    ground_state_shell = (4, 2)
