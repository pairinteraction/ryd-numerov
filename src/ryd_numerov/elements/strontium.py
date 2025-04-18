from abc import ABC
from typing import Optional

from ryd_numerov.elements.element import Element


class Strontium(ABC):  # noqa: B024
    _ionization_energy: tuple[float, Optional[float], str] = (5.694_84, 0.000_02, "eV")
    # https://webbook.nist.gov/cgi/inchi?ID=C7440246&Mask=20


class StrontiumSinglet(Strontium, Element):
    species = "Sr_singlet"
    s = 0
    ground_state_shell = (5, 0)


class StrontiumTriplet(Strontium, Element):
    species = "Sr_triplet"
    s = 1
    ground_state_shell = (4, 2)
