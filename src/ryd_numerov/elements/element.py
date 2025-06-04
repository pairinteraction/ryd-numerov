import inspect
from abc import ABC
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Optional, Union, overload

from ryd_numerov.units import ureg

if TYPE_CHECKING:
    from ryd_numerov.units import PintFloat

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
    ground_state_shell: ClassVar[tuple[int, int]]
    """Shell (n, l) describing the electronic ground state configuration."""
    _ionization_energy: tuple[float, Optional[float], str]
    """Ionization energy with uncertainty and unit: (value, uncertainty, unit)."""
    add_spin_orbit: ClassVar[bool] = True
    """Whether the default for this element is to add spin-orbit coupling to the Hamiltonian
    (mainly used for H_textbook)."""

    @classmethod
    @cache
    def from_species(cls, species: str) -> "Element":
        """Create an instance of the element class from the species string.

        Args:
            species: The species string (e.g. "Rb").

        Returns:
            An instance of the corresponding element class.

        """
        def get_concrete_subclasses(_cls: type[Element]) -> list[type[Element]]:
            subclasses = []
            for subclass in _cls.__subclasses__():
                if not inspect.isabstract(subclass) and hasattr(subclass, "species"):
                    subclasses.append(subclass)
                subclasses.extend(get_concrete_subclasses(subclass))
            return subclasses

        concrete_subclasses = get_concrete_subclasses(cls)
        for subclass in concrete_subclasses:
            if subclass.species == species:
                return subclass()
        raise ValueError(
            f"Unknown species: {species}. Available species: {[subclass.species for subclass in concrete_subclasses]}"
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
        if n < 1 or l < 0 or l >= n:
            raise ValueError(f"Invalid shell: (n={n}, l={l}). Must be n >= 1 and 0 <= l < n.")
        if (n, l) not in SORTED_SHELLS:
            return True
        return SORTED_SHELLS.index((n, l)) >= SORTED_SHELLS.index(self.ground_state_shell)

    @overload
    def get_ionization_energy(self, unit: None = None) -> "PintFloat": ...

    @overload
    def get_ionization_energy(self, unit: str) -> float: ...

    def get_ionization_energy(self, unit: Optional[str] = "hartree") -> Union["PintFloat", float]:
        """Return the ionization energy in the desired unit.

        Args:
            unit: Desired unit for the ionization energy. Default is atomic units "hartree".

        Returns:
            Ionization energy in the desired unit.

        """
        ionization_energy: PintFloat = ureg.Quantity(self._ionization_energy[0], self._ionization_energy[2])
        ionization_energy = ionization_energy.to("hartree", "spectroscopy")
        if unit is None:
            return ionization_energy
        if unit == "a.u.":
            return ionization_energy.magnitude
        return ionization_energy.to(unit, "spectroscopy").magnitude  # type: ignore [no-any-return]  # pint typing .to(unit)
