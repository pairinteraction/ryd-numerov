from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, get_args, overload

import numpy as np

from ryd_numerov.angular import AngularKetLS
from ryd_numerov.angular.utils import try_trivial_spin_addition
from ryd_numerov.elements.base_element import BaseElement
from ryd_numerov.radial import RadialState
from ryd_numerov.units import BaseQuantities, MatrixElementType, ureg

if TYPE_CHECKING:
    from typing_extensions import Self

    from ryd_numerov.angular import AngularKetBase
    from ryd_numerov.units import PintFloat


logger = logging.getLogger(__name__)
OPERATOR_TO_KS = {  # operator: (k_radial, k_angular)
    "MAGNETIC_DIPOLE": (0, 1),
    "ELECTRIC_DIPOLE": (1, 1),
    "ELECTRIC_QUADRUPOLE": (2, 2),
    "ELECTRIC_OCTUPOLE": (3, 3),
    "ELECTRIC_QUADRUPOLE_ZERO": (2, 0),
}


class RydbergStateBase(ABC):
    species: str

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def element(self) -> BaseElement:
        """The element of the Rydberg state."""
        if not hasattr(self, "_element"):
            self.create_element()
        return self._element

    def create_element(self, *, use_nist_data: bool = True) -> None:
        """Create the element for the Rydberg state."""
        if hasattr(self, "_element"):
            raise RuntimeError("The element was already created, you should not create it again.")
        self._element = BaseElement.from_species(self.species, use_nist_data=use_nist_data)

    @property
    @abstractmethod
    def radial(self) -> RadialState: ...

    @property
    @abstractmethod
    def angular(self) -> AngularKetBase: ...

    @abstractmethod
    def get_nu(self) -> float:
        """Get the effective principal quantum number nu (for alkali atoms also known as n*) for the Rydberg state."""

    @overload
    def get_energy(self, unit: None = None) -> PintFloat: ...

    @overload
    def get_energy(self, unit: str) -> float: ...

    def get_energy(self, unit: str | None = None) -> PintFloat | float:
        r"""Get the energy of the Rydberg state.

        The energy is defined as

        .. math::
            E = - \frac{1}{2} \frac{\mu}{\nu^2}

        where `\mu = R_M/R_\infty` is the reduced mass and `\nu` the effective principal quantum number.
        """
        nu = self.get_nu()
        energy_au = self.element.calc_energy_from_nu(nu)
        if unit == "a.u.":
            return energy_au
        energy: PintFloat = energy_au * BaseQuantities["ENERGY"]
        if unit is None:
            return energy
        return energy.to(unit, "spectroscopy").magnitude

    @overload
    def calc_reduced_matrix_element(self, other: Self, operator: MatrixElementType, unit: None = None) -> PintFloat: ...

    @overload
    def calc_reduced_matrix_element(self, other: Self, operator: MatrixElementType, unit: str) -> float: ...

    def calc_reduced_matrix_element(
        self, other: Self, operator: MatrixElementType, unit: str | None = None
    ) -> PintFloat | float:
        r"""Calculate the reduced matrix element.

        Calculate the reduced matrix element between self and other (ignoring m quantum numbers)

        .. math::
            \left\langle self || r^k_radial \hat{O}_{k_angular} || other \right\rangle

        where \hat{O}_{k_angular} is the operator of rank k_angular for which to calculate the matrix element.
        k_radial and k_angular are determined from the operator automatically.

        Args:
            other: The other Rydberg state for which to calculate the matrix element.
            operator: The operator for which to calculate the matrix element.
            unit: The unit to which to convert the radial matrix element.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.

        Returns:
            The reduced matrix element for the given operator.

        """
        if operator not in get_args(MatrixElementType):
            raise ValueError(f"Operator {operator} not supported, must be one of {get_args(MatrixElementType)}")

        k_radial, k_angular = OPERATOR_TO_KS[operator]
        radial_matrix_element = self.radial.calc_matrix_element(other.radial, k_radial)

        matrix_element: PintFloat
        if operator == "MAGNETIC_DIPOLE":
            # Magnetic dipole operator: mu = - mu_B (g_l <l_tot> + g_s <s_tot>)
            g_s = 2.0023192
            value_s_tot = self.angular.calc_reduced_matrix_element(other.angular, "s_tot", k_angular)
            g_l = 1
            value_l_tot = self.angular.calc_reduced_matrix_element(other.angular, "l_tot", k_angular)
            angular_matrix_element = g_s * value_s_tot + g_l * value_l_tot

            matrix_element = -ureg.Quantity(1, "bohr_magneton") * radial_matrix_element * angular_matrix_element
            # Note: we use the convention, that the magnetic dipole moments are given
            # as the same dimensionality as the Bohr magneton (mu = - mu_B (g_l l + g_s s_tot))
            # such that - mu * B (where the magnetic field B is given in dimension Tesla) is an energy

        elif operator in ["ELECTRIC_DIPOLE", "ELECTRIC_QUADRUPOLE", "ELECTRIC_OCTUPOLE", "ELECTRIC_QUADRUPOLE_ZERO"]:
            # Electric multipole operator: p_{k,q} = e r^k_radial * sqrt(4pi / (2k+1)) * Y_{k_angular,q}(\theta, phi)
            angular_matrix_element = self.angular.calc_reduced_matrix_element(other.angular, "SPHERICAL", k_angular)
            matrix_element = (
                ureg.Quantity(1, "e")
                * math.sqrt(4 * np.pi / (2 * k_angular + 1))
                * radial_matrix_element
                * angular_matrix_element
            )

        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")

        if unit == "a.u.":
            return matrix_element.to_base_units().magnitude
        if unit is None:
            return matrix_element
        return matrix_element.to(unit).magnitude

    @overload
    def calc_matrix_element(self, other: Self, operator: MatrixElementType, q: int) -> PintFloat: ...

    @overload
    def calc_matrix_element(self, other: Self, operator: MatrixElementType, q: int, unit: str) -> float: ...

    def calc_matrix_element(
        self, other: Self, operator: MatrixElementType, q: int, unit: str | None = None
    ) -> PintFloat | float:
        r"""Calculate the matrix element.

        Calculate the full matrix element between self and other,
        also considering the magnetic quantum numbers m of self and other.

        .. math::
            \left\langle self || r^k_radial \hat{O}_{k_angular} || other \right\rangle

        where \hat{O}_{k_angular} is the operator of rank k_angular for which to calculate the matrix element.
        k_radial and k_angular are determined from the operator automatically.

        Args:
            other: The other Rydberg state for which to calculate the matrix element.
            operator: The operator for which to calculate the matrix element.
            q: The component of the operator.
            unit: The unit to which to convert the radial matrix element.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.

        Returns:
            The matrix element for the given operator.

        """
        _k_radial, k_angular = OPERATOR_TO_KS[operator]
        prefactor = self.angular._calc_wigner_eckart_prefactor(other.angular, k_angular, q)  # noqa: SLF001
        reduced_matrix_element = self.calc_reduced_matrix_element(other, operator, unit)
        return prefactor * reduced_matrix_element


class RydbergStateAlkali(RydbergStateBase):
    """Create an Alkali Rydberg state, including the radial and angular states."""

    def __init__(
        self,
        species: str,
        n: int,
        l: int,
        j: float | None = None,
        m: float | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            l: Orbital angular momentum quantum number of the rydberg electron.
            j: Angular momentum quantum number of the rydberg electron.
            m: Total magnetic quantum number.
              Optional, only needed for concrete angular matrix elements.

        """
        self.species = species
        self.n = n
        self.l = l
        self.j = try_trivial_spin_addition(l, 0.5, j, "j")
        self.m = m

        element = BaseElement.from_species(species)
        if element.number_valence_electrons != 1:
            raise ValueError(f"The element {species} is not an alkali atom.")
        if not element.is_allowed_shell(n, l, s_tot=1 / 2):
            raise ValueError(f"The shell ({n=}, {l=}) is not allowed for the species {self.species}.")

    @cached_property
    def angular(self) -> AngularKetLS:
        """The angular/spin state of the Rydberg electron."""
        return AngularKetLS(l_r=self.l, j_tot=self.j, m=self.m, species=self.species)

    @cached_property
    def radial(self) -> RadialState:
        """The radial state of the Rydberg electron."""
        radial_state = RadialState(self.species, nu=self.get_nu(), l_r=self.l)
        radial_state.set_n_for_sanity_check(self.n)
        return radial_state

    def __repr__(self) -> str:
        species, n, l, j, m = self.species, self.n, self.l, self.j, self.m
        return f"{self.__class__.__name__}({species}, {n=}, {l=}, {j=}, {m=})"

    def get_nu(self) -> float:
        energy_au = self.element.calc_energy(self.n, self.l, self.j, s_tot=1 / 2, unit="a.u.")
        return self.element.calc_nu_from_energy(energy_au)


class RydbergStateAlkaliHyperfine(RydbergStateBase):
    """Create an Alkali Rydberg state, including the radial and angular states."""

    def __init__(
        self,
        species: str,
        n: int,
        l: int,
        j: float | None = None,
        f: float | None = None,
        m: float | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            l: Orbital angular momentum quantum number of the rydberg electron.
            j: Angular momentum quantum number of the rydberg electron.
            f: Total angular momentum quantum number of the atom (rydberg electron + core)
            m: Total magnetic quantum number.
              Optional, only needed for concrete angular matrix elements.

        """
        element = BaseElement.from_species(species)

        self.species = species
        self.n = n
        self.l = l
        self.j = try_trivial_spin_addition(l, 0.5, j, "j")
        self.f = try_trivial_spin_addition(self.j, element.i_c, f, "f")
        self.m = m

        if element.number_valence_electrons != 1:
            raise ValueError(f"The element {species} is not an alkali atom.")
        if not element.is_allowed_shell(n, l, s_tot=1 / 2):
            raise ValueError(f"The shell ({n=}, {l=}) is not allowed for the species {self.species}.")

    @cached_property
    def angular(self) -> AngularKetLS:
        """The angular/spin state of the Rydberg electron."""
        return AngularKetLS(l_r=self.l, j_tot=self.j, m=self.m, f_tot=self.f, species=self.species)

    @cached_property
    def radial(self) -> RadialState:
        """The radial state of the Rydberg electron."""
        radial_state = RadialState(self.species, nu=self.get_nu(), l_r=self.l)
        radial_state.set_n_for_sanity_check(self.n)
        return radial_state

    def __repr__(self) -> str:
        species, n, l, j, f, m = self.species, self.n, self.l, self.j, self.f, self.m
        return f"{self.__class__.__name__}({species}, {n=}, {l=}, {j=}, {f=}, {m=})"

    def get_nu(self) -> float:
        energy_au = self.element.calc_energy(self.n, self.l, self.j, s_tot=1 / 2, unit="a.u.")
        return self.element.calc_nu_from_energy(energy_au)


class RydbergStateAlkalineLS(RydbergStateBase):
    """Create an Alkaline Rydberg state, including the radial and angular states."""

    def __init__(
        self,
        species: str,
        n: int,
        l: int,
        s_tot: float,
        j_tot: float | None = None,
        m: float | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            l: Orbital angular momentum quantum number of the rydberg electron.
            s_tot: Total spin quantum number of all electrons.
            j_tot: Total angular momentum quantum number of all electrons.
            m: Total magnetic quantum number.
              Optional, only needed for concrete angular matrix elements.

        """
        self.species = species
        self.n = n
        self.l = l
        self.s_tot = s_tot
        self.j_tot = try_trivial_spin_addition(l, s_tot, j_tot, "j_tot")
        self.m = m

        element = BaseElement.from_species(species)
        if element.number_valence_electrons != 2:
            raise ValueError(f"The element {species} is not an alkaline atom.")
        if not element.is_allowed_shell(n, l, s_tot=s_tot):
            raise ValueError(f"The shell ({n=}, {l=}) is not allowed for the species {self.species}.")

    @cached_property
    def angular(self) -> AngularKetLS:
        """The angular/spin state of the Rydberg electron."""
        return AngularKetLS(l_r=self.l, s_tot=self.s_tot, j_tot=self.j_tot, m=self.m, species=self.species)

    @cached_property
    def radial(self) -> RadialState:
        """The radial state of the Rydberg electron."""
        radial_state = RadialState(self.species, nu=self.get_nu(), l_r=self.l)
        radial_state.set_n_for_sanity_check(self.n)
        return radial_state

    def __repr__(self) -> str:
        species, n, l, s_tot, j_tot, m = self.species, self.n, self.l, self.s_tot, self.j_tot, self.m
        return f"{self.__class__.__name__}({species}, {n=}, {l=}, {s_tot=}, {j_tot=}, {m=})"

    def get_nu(self) -> float:
        energy_au = self.element.calc_energy(self.n, self.l, self.j_tot, s_tot=self.s_tot, unit="a.u.")
        return self.element.calc_nu_from_energy(energy_au)
