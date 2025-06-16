import inspect
import logging
import re
from abc import ABC
from fractions import Fraction
from functools import cache, cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, Union, overload

import numpy as np

from ryd_numerov.units import ureg

if TYPE_CHECKING:
    from ryd_numerov.model.model import PotentialType
    from ryd_numerov.units import PintFloat

logger = logging.getLogger(__name__)


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
    _additional_allowed_shells: ClassVar[list[tuple[int, int]]] = []
    """Additional allowed shells (n, l), which (n, l) is smaller than the ground state shell."""

    _core_electron_configuration: ClassVar[str]
    """Electron configuration of the core electrons, e.g. 4p6 for Rb or 5s for Sr."""
    _ionization_energy: tuple[float, Optional[float], str]
    """Ionization energy with uncertainty and unit: (value, uncertainty, unit)."""

    # Parameters for the extended Rydberg Ritz formula, see calc_n_star
    _quantum_defects: ClassVar[dict[tuple[int, float], tuple[float, float, float, float, float]]] = {}
    """Dictionary containing the quantum defects for each (l, j) combination, i.e.
    _quantum_defects[(l,j)] = (d0, d2, d4, d6, d8)
    """

    _corrected_rydberg_constant: tuple[float, Optional[float], str]
    r"""Corrected Rydberg constant stored as (value, uncertainty, unit)"""

    potential_type_default: Optional["PotentialType"] = None
    """Default potential type to use for this element. If None, the potential type must be specified explicitly.
    In general, it looks like marinescu_1993 is better for alkali atoms, and fei_2009 is better for alkaline earth atoms
    """

    # Model Potential Parameters for marinescu_1993
    alpha_c_marinescu_1993: ClassVar[float]
    """Static dipole polarizability in atomic units (a.u.), used for the parametric model potential.
    See also: Phys. Rev. A 49, 982 (1994)
    """
    r_c_dict_marinescu_1993: ClassVar[dict[int, float]]
    """Cutoff radius {l: r_c} to truncate the unphysical short-range contribution of the polarization potential.
    See also: Phys. Rev. A 49, 982 (1994)
    """
    model_potential_parameter_marinescu_1993: ClassVar[dict[int, tuple[float, float, float, float]]]
    """Parameters {l: (a_1, a_2, a_3, a_4)} for the parametric model potential.
    See also: M. Marinescu, Phys. Rev. A 49, 982 (1994), https://journals.aps.org/pra/abstract/10.1103/PhysRevA.49.982
    """

    # Model Potential Parameters for fei_2009
    model_potential_parameter_fei_2009: tuple[float, float, float, float]
    """Parameters (delta, alpha, beta, gamma) for the new four-parameter potential, used in the model potential
    defined in: Y. Fei et al., Chin. Phys. B 18, 4349 (2009), https://iopscience.iop.org/article/10.1088/1674-1056/18/10/025
    """

    _nist_energy_levels_file: Optional[Path] = None
    """Path to the NIST energy levels file for this element.
    The file should be directly downloaded from https://physics.nist.gov/PhysRefData/ASD/levels_form.html
    in the 'Tab-delimited' format and in units of Hartree.
    """

    def __init__(self, use_nist_data: bool = True, *, nist_n_max: int = 15) -> None:
        """Initialize an element instance.

        Use this init method to set up additional properties and data for the element,
        like loading NIST energy levels from a file.

        Args:
            use_nist_data: Whether to use NIST data for this element. Default is True.
            nist_n_max: Maximum principal quantum number for which to load the NIST energy levels. Default is 15.

        """
        self._nist_energy_levels: dict[tuple[int, int, float], float] = {}
        self._nist_n_max = nist_n_max
        self.use_nist_data = use_nist_data
        if use_nist_data and self._nist_energy_levels_file is not None:
            self._setup_nist_energy_levels(self._nist_energy_levels_file)

    def _setup_nist_energy_levels(self, file: Path) -> None:  # noqa: C901, PLR0912
        """Set up NIST energy levels from a file.

        This method should be called in the constructor to load the NIST energy levels
        from the specified file. It reads the file and prepares the data for further use.

        Args:
            file: Path to the NIST energy levels file.
            n_max: Maximum principal quantum number for which to load the NIST energy levels.
                For large quantum numbers, the NIST data is not accurate enough
                (it does not even show fine structure splitting),
                so we limit the maximum principal quantum number to 15 by default.

        """
        if not file.exists():
            raise ValueError(f"NIST energy data file {file} does not exist.")

        header = file.read_text().splitlines()[0]
        if "Level (Hartree)" not in header:
            raise ValueError(
                f"NIST energy data file {file} not given in Hartree, please download the data in units of Hartree."
            )

        data = np.loadtxt(file, skiprows=1, dtype=str, quotechar='"', delimiter="\t")
        # data[i] := (Configuration, Term, J, Prefix, Energy, Suffix, Uncertainty, Reference)
        core_config_parts = convert_electron_configuration(self._core_electron_configuration)

        for row in data:
            if re.match(r"^([A-Z])", row[0]):
                # Skip rows, where the first column starts with an element symbol
                continue

            try:
                config_parts = convert_electron_configuration(row[0])
            except ValueError:
                # Skip rows with invalid electron configuration format
                # (they usually correspond to core configurations, that are not the ground state configuration)
                # e.g. strontium "4d.(2D<3/2>).4f"
                continue
            if sum(part[2] for part in config_parts) != sum(part[2] for part in core_config_parts) + 1:
                # Skip configurations, where the number of electrons does not match the core configuration + 1
                continue

            for part in core_config_parts:
                if part in config_parts:
                    config_parts.remove(part)
                elif (part[0], part[1], part[2] + 1) in config_parts:
                    config_parts.remove((part[0], part[1], part[2] + 1))
                    config_parts.append((part[0], part[1], 1))
                else:
                    break
            if sum(part[2] for part in config_parts) != 1:
                # Skip configurations, where the inner electrons are not in the ground state configuration
                continue
            n, l = config_parts[0][:2]

            multiplicity = int(row[1][0])
            if (multiplicity - 1) / 2 != self.s:
                continue

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
        concrete_subclasses = cls._get_concrete_subclasses()
        for subclass in concrete_subclasses:
            if subclass.species == species:
                return subclass(use_nist_data=use_nist_data)
        raise ValueError(
            f"Unknown species: {species}. Available species: {[subclass.species for subclass in concrete_subclasses]}"
        )

    @classmethod
    def _get_concrete_subclasses(cls) -> list[type["BaseElement"]]:
        subclasses = []
        for subclass in cls.__subclasses__():
            if not inspect.isabstract(subclass) and hasattr(subclass, "species"):
                subclasses.append(subclass)
            subclasses.extend(subclass._get_concrete_subclasses())  # noqa: SLF001
        return subclasses

    @classmethod
    def get_available_species(cls) -> list[str]:
        """Get a list of all available species in the library.

        This method returns a list of species strings for all concrete subclasses of BaseElement.

        Returns:
            List of species strings.

        """
        return sorted([subclass.species for subclass in cls._get_concrete_subclasses()])

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
        if (n, l) >= self.ground_state_shell:
            return True
        return (n, l) in self._additional_allowed_shells

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
        assert j % 1 == (l + self.s) % 1, f"j % 1 must be same as (l + s) % 1, (s={self.s}, l={l}, j={j})"
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
        if j % 1 != (l + self.s) % 1:
            raise ValueError(f"Invalid quantum numbers: (s={self.s}, l={l}, j={j})")
        if n <= self._nist_n_max and self.use_nist_data:
            if (n, l, j) in self._nist_energy_levels:
                energy_au = self._nist_energy_levels[(n, l, j)]
                energy_au -= self.get_ionization_energy("hartree")
            else:
                logger.debug(
                    "NIST energy levels for (n=%d, l=%d, j=%s) not found, using quantum defect theory.", n, l, j
                )
                energy_au = -0.5 * self.reduced_mass_factor / self.calc_n_star(n, l, j) ** 2
        else:
            energy_au = -0.5 * self.reduced_mass_factor / self.calc_n_star(n, l, j) ** 2
        energy: PintFloat = ureg.Quantity(energy_au, "hartree")
        if unit is None:
            return energy
        if unit == "a.u.":
            return energy.magnitude
        return energy.to(unit, "spectroscopy").magnitude  # type: ignore [no-any-return]  # pint typing .to(unit)


def convert_electron_configuration(config: str) -> list[tuple[int, int, int]]:
    """Convert an electron configuration string to a list of tuples [(n, l, number), ...].

    This means convert a string representing the outermost electrons
    like "4f14.6s" to [(4, 2, 14), (6, 0, 1)].
    """
    l_str2int = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "k": 7, "l": 8, "m": 9}
    parts = config.split(".")
    converted_parts = []
    for part in parts:
        match = re.match(r"^(\d+)([a-z])(\d*)$", part)
        if match is None:
            raise ValueError(f"Invalid configuration format: {config}.")
        n = int(match.group(1))
        l = l_str2int[match.group(2)]
        number = int(match.group(3)) if match.group(3) else 1
        converted_parts.append((n, l, number))

    return converted_parts
