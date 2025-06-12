import inspect
import re
from abc import ABC
from fractions import Fraction
from functools import cache, cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, Union, overload

import numpy as np

from ryd_numerov.units import ureg

if TYPE_CHECKING:
    from ryd_numerov.model.model_potential import ADDITIONAL_POTENTIALS
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


class BaseElement(ABC):
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
    _core_electron_configuration: ClassVar[str]
    """Electron configuration of the core electrons, e.g. 4p6 for Rb or 5s for Sr."""
    _ionization_energy: tuple[float, Optional[float], str]
    """Ionization energy with uncertainty and unit: (value, uncertainty, unit)."""

    additional_potentials_default: ClassVar[list["ADDITIONAL_POTENTIALS"]] = []

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

    _nist_energy_levels_file: Optional[Path] = None
    """Path to the NIST energy levels file for this element.
    The file should be directly downloaded from https://physics.nist.gov/PhysRefData/ASD/levels_form.html
    in the 'Tab-delimited' format and in units of Hartree.
    """

    def __init__(self, use_nist_data: bool = True) -> None:
        """Initialize an element instance.

        Use this init method to set up additional properties and data for the element,
        like loading NIST energy levels from a file.

        Args:
            use_nist_data: Whether to use NIST data for this element. Default is True.

        """
        self._nist_energy_levels: dict[tuple[int, int, float], float] = {}
        if use_nist_data and self._nist_energy_levels_file is not None:
            self._setup_nist_energy_levels(self._nist_energy_levels_file)

    def _setup_nist_energy_levels(self, file: Path) -> None:  # noqa: C901
        """Set up NIST energy levels from a file.

        This method should be called in the constructor to load the NIST energy levels
        from the specified file. It reads the file and prepares the data for further use.
        """
        if not file.exists():
            raise ValueError(f"NIST energy data file {file} does not exist.")

        header = file.read_text().splitlines()[0]
        if "Level (Hartree)" not in header:
            raise ValueError(
                f"NIST energy data file {file} not given in Hartree, please download the data in units of Hartree."
            )

        l_str2int = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "k": 7, "l": 8, "m": 9}

        data = np.loadtxt(file, skiprows=1, dtype=str, quotechar='"', delimiter="\t")
        # data[i] := (Configuration, Term, J, Prefix, Energy, Suffix, Uncertainty, Reference)

        for row in data:
            if re.match(r"^([A-Z])", row[0]):
                # Skip rows, where the first column starts with an element symbol
                continue

            config: str = row[0]
            parts = config.split(".")
            if self._core_electron_configuration not in parts[0]:
                continue  # Skip configurations, where the inner electrons are not in the ground state configuration

            multiplicity = int(row[1][0])
            if (multiplicity - 1) / 2 != self.s:
                continue

            match = None
            if len(parts) == 1:
                match = re.match(r"^(\d+)([a-z])2$", parts[0])
            elif len(parts) == 2:
                match = re.match(r"^(\d+)([a-z])$", parts[1])
            if match is None:
                raise ValueError(f"Invalid configuration format: {config}.")

            n = int(match.group(1))
            l = l_str2int[match.group(2)]

            j_list = [float(Fraction(j_str)) for j_str in row[2].split(",")]
            for j in j_list:
                energy = float(row[4])
                self._nist_energy_levels[(n, l, j)] = energy

        if len(self._nist_energy_levels) == 0:
            raise ValueError(f"No NIST energy levels found for element {self.species} in file {file}.")

    @classmethod
    @cache
    def from_species(cls, species: str, use_nist_data: bool = True) -> "BaseElement":
        """Create an instance of the element class from the species string.

        This method searches through all subclasses of BaseElement until it finds one with a matching species attribute.
        This approach allows for easy extension of the library with new elements.
        A user can even subclass BaseElement in his code (without modifying the ryd-numerov library),
        e.g. `class CustomRubidium(BaseElement): species = "Custom_Rb" ...`
        and then use the new element by calling RydbergState("Custom_Rb", ...)

        Args:
            species: The species string (e.g. "Rb").
            use_nist_data: Whether to use NIST data for this element. Default is True.

        Returns:
            An instance of the corresponding element class.

        """

        def get_concrete_subclasses(_cls: type["BaseElement"]) -> list[type["BaseElement"]]:
            subclasses = []
            for subclass in _cls.__subclasses__():
                if not inspect.isabstract(subclass) and hasattr(subclass, "species"):
                    subclasses.append(subclass)
                subclasses.extend(get_concrete_subclasses(subclass))
            return subclasses

        concrete_subclasses = get_concrete_subclasses(cls)
        for subclass in concrete_subclasses:
            if subclass.species == species:
                return subclass(use_nist_data=use_nist_data)
        raise ValueError(
            f"Unknown species: {species}. Available species: {[subclass.species for subclass in concrete_subclasses]}"
        )

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

    @cached_property  # don't remove this caching without benchmarking it!!!
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
        if (n, l, j) in self._nist_energy_levels:
            energy_au = self._nist_energy_levels[(n, l, j)]
            energy_au -= self.get_ionization_energy("hartree")
        else:
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
