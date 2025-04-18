from abc import ABC
from functools import cache
from typing import ClassVar, Union

from ryd_numerov.units import ureg

# List of energetically sorted shells
SORTED_SHELLS = [  # (n, l)
    (1, 0),  # H
    (2, 0),  # Li, Be
    (2, 1),
    (3, 0),  # Na, Mg
    (3, 1),
    (4, 0),  # K, Ca
    (3, 2),
    (4, 1),
    (5, 0),  # Rb, Sr
    (4, 2),
    (5, 1),
    (6, 0),  # Cs, Ba
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),  # Fr, Ra
    (5, 3),
    (6, 2),
    (7, 1),
    (8, 0),
]


class Element(ABC):
    """Abstract base class for all elements.

    For the electronic ground state configurations and sorted shells,
    see e.g. https://www.webelements.com/atoms.html

    """

    species: ClassVar[str]
    """Atomic species."""
    s: ClassVar[Union[int, float]]
    """Total spin quantum number."""
    ground_state_shell: ClassVar[tuple[int, int]]  # (n, l)
    """Shell (n, l) describing the electronic ground state configuration."""
    ionization_energy_ghz: float
    """Ionization energy in GHz."""

    @classmethod
    @cache
    def from_species(cls, species: str) -> "Element":
        """Create an instance of the element class from the species string.

        Args:
            species: The species string (e.g. "Rb").

        Returns:
            An instance of the corresponding element class.

        """
        for subclass in cls.__subclasses__():
            if subclass.species == species:
                return subclass()
        raise ValueError(
            f"Unknown species: {species}. Available species: {[subclass.species for subclass in cls.__subclasses__()]}"
        )

    @property
    def is_alkali(self) -> bool:
        """Check if the element is an alkali metal."""
        return self.s == 1 / 2

    def is_allowed_shell(self, n: int, l: int) -> bool:
        """Check if the quantum numbers describe an allowed shell.

        I.e. whether the shell is above the ground state shell.

        Args:
            n: Principal quantum number
            l: Orbital angular momentum quantum number

        Returns:
            True if the quantum numbers specify a shell equal to or above the ground state shell, False otherwise.

        """
        if (n, l) not in SORTED_SHELLS:
            return True
        if self.species in ["Sr_triplet"] and (n, l) == (4, 2):  # Sr_triplet has a special case
            return True
        gs_id = SORTED_SHELLS.index(self.ground_state_shell)
        state_id = SORTED_SHELLS.index((n, l))
        return state_id >= gs_id

    def get_ionization_energy(self, unit: str = "hartree") -> float:
        """Return the ionization energy in the desired unit.

        Args:
            unit: Desired unit for the ionization energy. Default is atomic units "hartree".

        Returns:
            Ionization energy in the desired unit.

        """
        return ureg.Quantity(self.ionization_energy_ghz, "GHz").to(unit, "spectroscopy").magnitude  # type: ignore [no-any-return]  # pint typing .to(unit)
