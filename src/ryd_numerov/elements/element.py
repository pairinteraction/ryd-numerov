import inspect
from abc import ABC
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Optional, Union, overload

import numpy as np

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
    Z: ClassVar[int]
    """Atomic number of the element."""
    s: ClassVar[Union[int, float]]
    """Total spin quantum number."""
    ground_state_shell: ClassVar[tuple[int, int]]
    """Shell (n, l) describing the electronic ground state configuration."""
    _ionization_energy: tuple[float, Optional[float], str]
    """Ionization energy with uncertainty and unit: (value, uncertainty, unit)."""
    add_spin_orbit: ClassVar[bool] = True
    """Whether the default for this element is to add spin-orbit coupling to the Hamiltonian
    (mainly used for H_textbook)."""

    # Parameters for the extended Rydberg Ritz formula, see calc_n_star
    _quantum_defects: ClassVar[dict[tuple[int, float], tuple[float, float, float, float, float]]] = {}
    """Dictionary containing the quantum defects for each (l, j) combination, i.e.
    _quantum_defects[(l,j)] = (d0, d2, d4, d6, d8)
    """

    _corrected_rydberg_constant: tuple[float, Optional[float], str]
    r"""Corrected Rydberg constant stored as (value, uncertainty, unit)"""

    alpha_c: ClassVar[float] = 0
    """Static dipole polarizability in atomic units (a.u.), used for the parametric model potential.
    See also: Phys. Rev. A 49, 982 (1994)
    """
    _r_c_dict: ClassVar[dict[int, float]] = {0: np.inf}
    """Cutoff radius {l: r_c} to truncate the unphysical short-range contribution of the polarization potential.
    See also: Phys. Rev. A 49, 982 (1994)
    """
    _parametric_model_potential_parameters: ClassVar[dict[int, tuple[float, float, float, float]]] = {}
    """Parameters {l: (a_1, a_2, a_3, a_4)} for the parametric model potential.
    See also: Phys. Rev. A 49, 982 (1994)
    """

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

    @overload
    def get_corrected_rydberg_constant(self, unit: None = None) -> "PintFloat": ...

    @overload
    def get_corrected_rydberg_constant(self, unit: str) -> float: ...

    def get_corrected_rydberg_constant(self, unit: Optional[str] = "hartree") -> Union["PintFloat", float]:
        r"""Return the corrected Rydberg constant in the desired unit.

        The corrected Rydberg constant is defined as

        .. math::
            R_M = R_\infty * \frac{m_{Core}}{m_{Core} + m_e}

        where :math:`R_\infty` is the Rydberg constant for infinite nuclear mass,
        :math:`m_{Core}` is the mass of the core,
        and :math:`m_e` is the mass of the electron.

        Args:
            unit: Desired unit for the corrected Rydberg constant. Default is atomic units "hartree".

        Returns:
            Corrected Rydberg constant in the desired unit.

        """
        corrected_rydberg_constant: PintFloat = ureg.Quantity(
            self._corrected_rydberg_constant[0], self._corrected_rydberg_constant[2]
        )
        corrected_rydberg_constant = corrected_rydberg_constant.to("hartree", "spectroscopy")
        if unit is None:
            return corrected_rydberg_constant
        if unit == "a.u.":
            return corrected_rydberg_constant.magnitude
        return corrected_rydberg_constant.to(unit, "spectroscopy").magnitude  # type: ignore [no-any-return]  # pint typing .to(unit)

    @property
    def reduced_mass_factor(self) -> float:
        r"""The reduced mass factor \mu.

        The reduced mass factor

        .. math::
            \mu = \frac{m_{Core}}{m_{Core} + m_e}

        calculated via the corrected Rydberg constant

        .. math::
            \mu = \frac{R_M}{R_\infty}

        """
        return (  # type: ignore [no-any-return]  # pint typing .to(unit)
            self.get_corrected_rydberg_constant("hartree")
            / ureg.Quantity(1, "rydberg_constant").to("hartree", "spectroscopy").magnitude
        )

    def calc_n_star(self, n: int, l: int, j: float) -> float:
        r"""Calculate the effective principal quantum number for the given n, l and j.

        The effective principal quantum number in quantum defect theory
        is defined as series expansion :math:`n^* = n - \delta_{lj}(n)`
        where

        .. math::
            \delta_{lj}(n) = d0_{lj} + d2_{lj} / [n - d0_{lj}(n)]^2 + d4_{lj} / [n - \delta_{lj}(n)]^4 + ...

        References:
            - On a New Law of Series Spectra, Ritz; DOI: 10.1086/141591, https://ui.adsabs.harvard.edu/abs/1908ApJ....28..237R/abstract
            - Rydberg atoms, Gallagher; DOI: 10.1088/0034-4885/51/2/001, (Eq. 16.19)

        """
        assert j % 1 in [0, 0.5], f"j must be integer or half-integer, but is {j}"
        d0, d2, d4, d6, d8 = self._quantum_defects.get((l, j), (0, 0, 0, 0, 0))
        delta_nlj = d0 + d2 / (n - d0) ** 2 + d4 / (n - d0) ** 4 + d6 / (n - d0) ** 6 + d8 / (n - d0) ** 8
        return n - delta_nlj

    @overload
    def calc_energy(self, n: int, l: int, j: float, unit: None = None) -> "PintFloat": ...

    @overload
    def calc_energy(self, n: int, l: int, j: float, unit: str) -> float: ...

    def calc_energy(self, n: int, l: int, j: float, unit: Optional[str] = "hartree") -> Union["PintFloat", float]:
        r"""Calculate the energy of a Rydberg state with for the given n, l and j.

        is the quantum defect. The energy of the Rydberg state is then given by

        .. math::
            E_{nlj} / E_H = -\frac{1}{2} \frac{Ry}{Ry_\infty} \frac{1}{n^*}

        where :math:`E_H` is the Hartree energy (the atomic unit of energy).
        """
        energy_au = -0.5 * self.reduced_mass_factor / self.calc_n_star(n, l, j) ** 2
        energy: PintFloat = ureg.Quantity(energy_au, "hartree")
        if unit is None:
            return energy
        if unit == "a.u.":
            return energy.magnitude
        return energy.to(unit, "spectroscopy").magnitude  # type: ignore [no-any-return]  # pint typing .to(unit)

    def get_parametric_model_potential_parameters(self, l: int) -> tuple[float, float, float, float]:
        """Get the parameters for the parametric model potential for the given principal quantum number n.

        Args:
            l: Orbital angular momentum quantum number.

        Returns:
            Parameters (a_1, a_2, a_3, a_4) for the parametric model potential.

        """
        if len(self._parametric_model_potential_parameters) == 0:
            raise ValueError("No parametric model potential parameters defined for this element.")
        if l in self._parametric_model_potential_parameters:
            return self._parametric_model_potential_parameters[l]
        max_l = max(self._parametric_model_potential_parameters.keys())
        return self._parametric_model_potential_parameters[max_l]

    def get_r_c(self, l: int) -> float:
        """Get the cutoff radius for the polarization potential for the given orbital angular momentum quantum number l.

        Args:
            l: Orbital angular momentum quantum number.

        Returns:
            Cutoff radius r_c for the polarization potential.

        """
        if l in self._r_c_dict:
            return self._r_c_dict[l]
        max_l = max(self._r_c_dict.keys())
        return self._r_c_dict[max_l]
