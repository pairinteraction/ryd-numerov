from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, get_args, overload

from ryd_numerov.elements.base_element import BaseElement
from ryd_numerov.radial_state import RadialState
from ryd_numerov.spin_state import SpinStateLS, _try_trivial_spin_addition
from ryd_numerov.units import BaseQuantities, OperatorType, ureg

if TYPE_CHECKING:
    from typing_extensions import Self

    from ryd_numerov.spin_state import SpinStateBase
    from ryd_numerov.units import PintFloat


logger = logging.getLogger(__name__)


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
    def radial_state(self) -> RadialState: ...

    @property
    @abstractmethod
    def spin_state(self) -> SpinStateBase: ...

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
    def calc_matrix_element(
        self, other: Self, operator: OperatorType, k_radial: int, k_angular: int, q: int
    ) -> PintFloat: ...

    @overload
    def calc_matrix_element(
        self, other: Self, operator: OperatorType, k_radial: int, k_angular: int, q: int, unit: str
    ) -> float: ...

    def calc_matrix_element(
        self, other: Self, operator: OperatorType, k_radial: int, k_angular: int, q: int, unit: str | None = None
    ) -> PintFloat | float:
        r"""Calculate the matrix element.

        Calculate the matrix element between two Rydberg states
        \ket{self}=\ket{n',l',j_tot',s_tot',m'} and \ket{other}= \ket{n,l,j_tot,s_tot,m}.

        .. math::
            \langle n,l,j_tot,s_tot,m | r^k_radial \hat{O}_{k_angular,q} | n',l',j_tot',s_tot',m' \rangle

        where \hat{O}_{k_angular,q} is the operators of rank k_angular and component q,
        for which to calculate the matrix element.

        Args:
            other: The other Rydberg state \ket{n,l,j_tot,s_tot,m} to which to calculate the matrix element.
            operator: The operator type for which to calculate the matrix element.
                Can be one of "MAGNETIC", "ELECTRIC", "SPHERICAL".
            k_radial: The radial matrix element power k.
            k_angular: The rank of the angular operator.
            q: The component of the angular operator.
            unit: The unit to which to convert the radial matrix element.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.

        Returns:
            The matrix element for the given operator.

        """
        assert operator in get_args(OperatorType), (
            f"Operator {operator} not supported, must be one of {get_args(OperatorType)}"
        )
        radial_matrix_element_au = self.radial_state.calc_matrix_element(other.radial_state, k_radial, unit="a.u.")
        angular_matrix_element_au = self.spin_state.calc_matrix_element(other.spin_state, operator, k_angular, q)
        matrix_element_au = radial_matrix_element_au * angular_matrix_element_au

        if operator == "MAGNETIC":
            matrix_element_au *= -0.5  # - mu_B in atomic units
        elif operator == "ELECTRIC":
            pass  # e in atomic units is 1

        if unit == "a.u.":
            return matrix_element_au

        matrix_element: PintFloat = matrix_element_au * (ureg.Quantity(1, "a0") ** k_radial)
        if operator == "ELECTRIC":
            matrix_element *= ureg.Quantity(1, "e")
        elif operator == "MAGNETIC":
            # 2 mu_B = hbar e / m_e = 1 a.u. = 1 atomic_unit_of_current * bohr ** 2
            # Note: we use the convention, that the magnetic dipole moments are given
            # as the same dimensionality as the Bohr magneton (mu = - mu_B (g_l l + g_s s_tot))
            # such that - mu * B (where the magnetic field B is given in dimension Tesla) is an energy
            matrix_element *= ureg.Quantity(2, "bohr_magneton")

        if unit is None:
            return matrix_element
        return matrix_element.to(unit).magnitude


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
        self.j = _try_trivial_spin_addition(l, 0.5, j, "j")
        self.m = m

        element = BaseElement.from_species(species)
        if element.number_valence_electrons != 1:
            raise ValueError(f"The element {species} is not an alkali atom.")
        if not element.is_allowed_shell(n, l, s_tot=1 / 2):
            raise ValueError(f"The shell ({n=}, {l=}) is not allowed for the species {self.species}.")

    @cached_property
    def spin_state(self) -> SpinStateLS:
        """The spin state of the Rydberg electron."""
        return SpinStateLS(l_r=self.l, j_tot=self.j, m=self.m, species=self.species)

    @cached_property
    def radial_state(self) -> RadialState:
        """The radial state of the Rydberg electron."""
        return RadialState(self.species, n=self.n, l_r=self.l, nu=self.get_nu())

    def __repr__(self) -> str:
        species, n, l, j, m = self.species, self.n, self.l, self.j, self.m
        return f"{self.__class__.__name__}({species}, {n=}, {l=}, {j=}, {m=})"

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
        self.j_tot = _try_trivial_spin_addition(l, s_tot, j_tot, "j_tot")
        self.m = m

        element = BaseElement.from_species(species)
        if element.number_valence_electrons != 2:
            raise ValueError(f"The element {species} is not an alkaline atom.")
        if not element.is_allowed_shell(n, l, s_tot=s_tot):
            raise ValueError(f"The shell ({n=}, {l=}) is not allowed for the species {self.species}.")

    @cached_property
    def spin_state(self) -> SpinStateLS:
        """The spin state of the Rydberg electron."""
        return SpinStateLS(l_r=self.l, s_tot=self.s_tot, j_tot=self.j_tot, m=self.m, species=self.species)

    @cached_property
    def radial_state(self) -> RadialState:
        """The radial state of the Rydberg electron."""
        return RadialState(self.species, n=self.n, l_r=self.l, nu=self.get_nu())

    def __repr__(self) -> str:
        species, n, l, s_tot, j_tot, m = self.species, self.n, self.l, self.s_tot, self.j_tot, self.m
        return f"{self.__class__.__name__}({species}, {n=}, {l=}, {s_tot=}, {j_tot=}, {m=})"

    def get_nu(self) -> float:
        energy_au = self.element.calc_energy(self.n, self.l, self.j_tot, s_tot=self.s_tot, unit="a.u.")
        return self.element.calc_nu_from_energy(energy_au)
