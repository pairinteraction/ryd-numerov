import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Optional, Union, get_args, overload

import numpy as np
from scipy.special import exprel

from ryd_numerov.angular import calc_angular_matrix_element
from ryd_numerov.elements import BaseElement
from ryd_numerov.model import Model
from ryd_numerov.radial import (
    Grid,
    Wavefunction,
    WavefunctionNumerov,
    WavefunctionWhittaker,
    calc_radial_matrix_element,
)
from ryd_numerov.units import BaseQuantities, OperatorType, ureg

if TYPE_CHECKING:
    from typing_extensions import Self

    from ryd_numerov.model.model import PotentialType
    from ryd_numerov.radial.wavefunction import WavefunctionSignConvention
    from ryd_numerov.units import NDArray, PintArray, PintFloat


logger = logging.getLogger(__name__)

TransitionRateMethod = Literal["exact", "approximation"]


class _CommonRydbergState(ABC):
    species: str
    n: Optional[int]
    l: int

    @overload
    def get_energy(self, unit: None = None) -> "PintFloat": ...

    @overload
    def get_energy(self, unit: str) -> float: ...

    @abstractmethod
    def get_energy(self, unit: Optional[str] = None) -> Union["PintFloat", float]: ...

    def get_n_star(self) -> float:
        r"""Calculate the effective quantum number n* for the Rydberg state.

        We define n* as

        .. math::
            n^* = \sqrt{-\frac{1}{2} \frac{\mu}{E} }

        where `\mu = R_M/R_\infty` is the reduced mass and `E` the energy of the state.

        """
        energy_au = self.get_energy("a.u.")
        return np.sqrt(-0.5 * self.element.reduced_mass_factor / energy_au)  # type: ignore [no-any-return] # numpy

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
    def model(self) -> Model:
        if not hasattr(self, "_model"):
            self.create_model()
        return self._model

    def create_model(self, potential_type: Optional["PotentialType"] = None) -> None:
        """Create the model for the Rydberg state.

        Args:
            potential_type: Which potential to use for the model.

        """
        if hasattr(self, "_model"):
            raise RuntimeError("The model was already created, you should not create it again.")

        self._model = Model(
            self.element,
            self.l,
            potential_type,
        )

    @property
    def grid(self) -> Grid:
        """The grid object for the integration of the radial Schrödinger equation."""
        if not hasattr(self, "_grid"):
            self.create_grid()
        return self._grid

    @property
    def z_list(self) -> "NDArray":
        """The list of z values for the grid."""
        return self.grid.z_list

    def create_grid(
        self,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        dz: float = 1e-2,
    ) -> None:
        """Create the grid object for the integration of the radial Schrödinger equation.

        Args:
            x_min: The minimum value of the radial coordinate in dimensionless units (x = r/a_0).
                Default: Automatically calculate sensible value.
            x_max: The maximum value of the radial coordinate in dimensionless units (x = r/a_0).
                Default: Automatically calculate sensible value.
            dz: The step size of the integration (z = r/a_0). Default: 1e-2.

        """
        if hasattr(self, "_grid"):
            raise RuntimeError("The grid was already created, you should not create it again.")

        if x_min is None:
            # we set z_min explicitly too small,
            # since the integration will automatically stop after the turning point,
            # and as soon as the wavefunction is close to zero
            if self.l <= 10:
                z_min = 0.0
            else:
                z_min = self.model.calc_turning_point_z(self.get_energy("a.u."))
                z_min = np.sqrt(0.5) * z_min - 3  # see also compare_z_min_cutoff.ipynb
        else:
            z_min = np.sqrt(x_min)
        # Since the potential diverges at z=0 we set the minimum z_min to dz
        z_min = max(z_min, dz)

        if x_max is None:
            n = self.n if self.n is not None else self.get_n_star() + 5
            # This is an empirical formula for the maximum value of the radial coordinate
            # it takes into account that for large n but small l the wavefunction is very extended
            x_max = 2 * n * (n + 15 + (n - self.l) / 4)
        z_max = np.sqrt(x_max)

        self._grid = Grid(z_min, z_max, dz)

    @property
    def wavefunction(self) -> Wavefunction:
        if not hasattr(self, "_wavefunction"):
            self._wavefunction: Wavefunction
            self.create_wavefunction()
        return self._wavefunction

    @property
    def w_list(self) -> "NDArray":
        """The list of w values for the wavefunction."""
        return self.wavefunction.w_list

    @overload
    def create_wavefunction(self, *, sign_convention: "WavefunctionSignConvention" = None) -> None: ...

    @overload
    def create_wavefunction(
        self,
        method: Literal["numerov"],
        sign_convention: "WavefunctionSignConvention" = None,
        *,
        run_backward: bool = True,
        w0: float = 1e-10,
        _use_njit: bool = True,
    ) -> None: ...

    @overload
    def create_wavefunction(
        self, method: Literal["whittaker"], sign_convention: "WavefunctionSignConvention" = None
    ) -> None: ...

    def create_wavefunction(
        self,
        method: Literal["numerov", "whittaker"] = "numerov",
        sign_convention: "WavefunctionSignConvention" = None,
        *,
        run_backward: bool = True,
        w0: float = 1e-10,
        _use_njit: bool = True,
    ) -> None:
        if hasattr(self, "_wavefunction"):
            raise RuntimeError("The wavefunction was already created, you should not create it again.")

        if method == "numerov":
            self._wavefunction = WavefunctionNumerov(self, self.grid, self.model)
            self._wavefunction.integrate(run_backward, w0, _use_njit=_use_njit)
        elif method == "whittaker":
            self._wavefunction = WavefunctionWhittaker(self, self.grid)
            self._wavefunction.integrate()

        if sign_convention is None:
            sign_convention = "n_l_1" if self.element.number_valence_electrons == 1 else "positive_at_outer_bound"

        self._wavefunction.apply_sign_convention(sign_convention)
        self._grid = self._wavefunction.grid

    @overload
    def calc_radial_matrix_element(self, other: "Self", k_radial: int) -> "PintFloat": ...

    @overload
    def calc_radial_matrix_element(self, other: "Self", k_radial: int, unit: str) -> float: ...

    def calc_radial_matrix_element(
        self, other: "Self", k_radial: int, unit: Optional[str] = None
    ) -> Union["PintFloat", float]:
        radial_matrix_element_au = calc_radial_matrix_element(self, other, k_radial)
        if unit == "a.u.":
            return radial_matrix_element_au
        radial_matrix_element: PintFloat = radial_matrix_element_au * BaseQuantities["RADIAL_MATRIX_ELEMENT"]
        if unit is None:
            return radial_matrix_element
        return radial_matrix_element.to(unit).magnitude


class RydbergStateSQDT(_CommonRydbergState):
    r"""Create a Rydberg state, for which the radial Schrödinger equation is solved using the Numerov method.

    This class is meant as single-channel quantum defect theory description,
    for which all quantum numbers are well defined.

    Integrate the radial Schrödinger equation for the Rydberg state using the Numerov method.

    We solve the radial dimensionless Schrödinger equation for the Rydberg state

    .. math::
        \frac{d^2}{dx^2} u(x) = - \left[ E - V_{eff}(x) \right] u(x)

    using the Numerov method, see `integration.run_numerov_integration`.

    """

    n: int

    def __init__(
        self,
        species: str,
        n: int,
        l: int,
        j_tot: Optional[float] = None,
        s_tot: Optional[float] = None,
        m: Optional[float] = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            l: Orbital angular momentum quantum number of the rydberg electron.
            j_tot: Angular momentum quantum number of the rydberg electron.
            s_tot: Total spin quantum number
              Optional, only needed for alkaline earth atoms, where it can be 0 (singlet) or 1 (triplet).
            m: Total magnetic quantum number.
              Optional, only needed for concrete angular matrix elements.

        """
        self.species = species

        self.n = n
        self.l = l

        self.s_tot: float = s_tot  # type: ignore [assignment] # assert not None below
        if s_tot is None and self.element.number_valence_electrons == 1:
            self.s_tot = 1 / 2
        assert self.s_tot is not None, "s_tot must be set"

        self.j_tot: float = j_tot  # type: ignore [assignment] # assert not None below
        if j_tot is None:
            if self.l == 0:
                self.j_tot = self.s_tot
            elif self.s_tot == 0:
                self.j_tot = self.l
        assert self.j_tot is not None, "j_tot must be set"

        self.m = m

        self.sanity_check()

    def __repr__(self) -> str:
        species, n, l, j_tot, s_tot, m = self.species, self.n, self.l, self.j_tot, self.s_tot, self.m
        return f"{self.__class__.__name__}({species}, {n=}, {l=}, {j_tot=}, {s_tot=}, {m=})"

    def __str__(self) -> str:
        return self.get_label("ket")

    def copy(self) -> "Self":
        """Create a copy of the Rydberg state."""
        return self.__class__(self.species, n=self.n, l=self.l, j_tot=self.j_tot, s_tot=self.s_tot, m=self.m)

    def get_label(self, fmt: Literal["raw", "ket", "bra"]) -> str:
        """Label representing the ket.

        Args:
            fmt: The format of the label, i.e. whether to return the raw label, or the label in ket or bra notation.

        Returns:
            The label of the ket in the given format.

        """
        l_dict = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}
        l_str = l_dict.get(self.l, self.l)
        j_str = f"{self.j_tot:.1f}" if self.j_tot % 1 != 0 else f"{int(self.j_tot)}"

        raw = f"{self.species}:{self.n},{l_str}_{j_str}"
        if self.s_tot != 1 / 2:
            raw += f",s={self.s_tot}"
        if self.m is not None:
            raw += f",m={self.m}"

        if fmt == "raw":
            return raw
        if fmt == "ket":
            return f"|{raw}⟩"
        if fmt == "bra":
            return f"⟨{raw}|"
        raise ValueError(f"Unknown fmt {fmt}")

    def sanity_check(self) -> None:  # noqa: C901
        """Check that the quantum numbers are valid."""
        msgs: list[str] = []
        n, l, j_tot, s_tot, m = self.n, self.l, self.j_tot, self.s_tot, self.m

        if not isinstance(n, (int, np.integer)):
            msgs.append(f"n must be an integer, but {n=}")
        if not n >= 1:
            msgs.append(f"n must be larger than 0, but is {n=}")

        if not isinstance(l, (int, np.integer)):
            msgs.append(f"l must be an integer, but {l=}")
        if not 0 <= l <= n - 1:
            msgs.append(f"l must be between 0 and n - 1, but {l=}, {n=}")

        if not abs(l - s_tot) <= j_tot <= l + s_tot:
            msgs.append(f"j_tot must be between |l - s_tot| and |l + s_tot|, but {l=}, {s_tot=}, {j_tot=}")

        if m is not None and not -j_tot <= m <= j_tot:
            msgs.append(f"m must be between -j_tot and j_tot, but {j_tot=}, {m=}")

        if self.element.number_valence_electrons == 1:
            msgs += self._sanity_check_alkali()
        elif self.element.number_valence_electrons == 2:
            msgs += self._sanity_check_alkaline_earth()

        if not self.element.is_allowed_shell(n, l, s_tot):
            msgs.append(f"The shell ({n=}, {l=}) is not allowed for the species {self.species}.")

        for msg in msgs:
            logger.error(msg)
        if msgs:
            raise ValueError(f"Invalid Rydberg state {self!r}")

    def _sanity_check_alkali(self) -> list[str]:
        msgs: list[str] = []
        j_tot, s_tot, m = self.j_tot, self.s_tot, self.m

        if s_tot != 1 / 2:
            msgs.append("Spin quantum number s_tot must be 1 / 2 for alkali atoms.")
        if j_tot % 1 != 1 / 2:
            msgs.append("Total angular momentum quantum number j_tot must be half-int for alkali atoms.")
        if m is not None and m % 1 != 1 / 2:
            msgs.append("Total magnetic quantum number m must be half-int for alkali atoms.")

        return msgs

    def _sanity_check_alkaline_earth(self) -> list[str]:
        msgs: list[str] = []
        j_tot, s_tot, m = self.j_tot, self.s_tot, self.m

        if s_tot not in [0, 1]:
            msgs.append("Spin quantum number s_tot must be given and 0 or 1 for alkaline earth atoms.")
        if j_tot % 1 != 0:
            msgs.append("Total angular momentum quantum number j_tot must be integer for alkaline earth atoms.")
        if m is not None and m % 1 != 0:
            msgs.append("Total magnetic quantum number m must be integer for alkaline earth atoms.")

        return msgs

    @overload
    def get_energy(self, unit: None = None) -> "PintFloat": ...

    @overload
    def get_energy(self, unit: str) -> float: ...

    def get_energy(self, unit: Optional[str] = None) -> Union["PintFloat", float]:
        energy_au = self.element.calc_energy(self.n, self.l, self.j_tot, self.s_tot, unit="a.u.")
        if unit == "a.u.":
            return energy_au
        energy: PintFloat = energy_au * BaseQuantities["ENERGY"]
        if unit is None:
            return energy
        return energy.to(unit, "spectroscopy").magnitude

    def calc_angular_matrix_element(self, other: "Self", operator: "OperatorType", k_angular: int, q: int) -> float:
        """Calculate the dimensionless angular matrix element."""
        if (self.s_tot is None or self.l is None or self.j_tot is None or self.m is None) or (
            other.s_tot is None or other.l is None or other.j_tot is None or other.m is None
        ):
            raise ValueError("l, j_tot, s_tot and m must be set to calculate the angular matrix element.")

        return calc_angular_matrix_element(
            self.s_tot, self.l, self.j_tot, self.m, other.s_tot, other.l, other.j_tot, other.m, operator, k_angular, q
        )

    @overload
    def calc_matrix_element(
        self, other: "Self", operator: "OperatorType", k_radial: int, k_angular: int, q: int
    ) -> "PintFloat": ...

    @overload
    def calc_matrix_element(
        self, other: "Self", operator: "OperatorType", k_radial: int, k_angular: int, q: int, unit: str
    ) -> float: ...

    def calc_matrix_element(
        self, other: "Self", operator: "OperatorType", k_radial: int, k_angular: int, q: int, unit: Optional[str] = None
    ) -> Union["PintFloat", float]:
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
        radial_matrix_element_au = self.calc_radial_matrix_element(other, k_radial, unit="a.u.")
        angular_matrix_element_au = self.calc_angular_matrix_element(other, operator, k_angular, q)
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

    @overload
    def get_spontaneous_transition_rates(
        self, *, method: TransitionRateMethod = "exact"
    ) -> tuple[list["Self"], "PintArray"]: ...

    @overload
    def get_spontaneous_transition_rates(
        self, unit: str, method: TransitionRateMethod = "exact"
    ) -> tuple[list["Self"], "NDArray"]: ...

    def get_spontaneous_transition_rates(
        self,
        unit: Optional[str] = None,
        method: TransitionRateMethod = "exact",
    ) -> tuple[list["Self"], Union["PintArray", "NDArray"]]:
        """Calculate the spontaneous transition rates for the Rydberg state.

        The spontaneous transition rates are given by the Einstein A coefficients.

        Args:
            unit: The unit to which to convert the result to.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.
            method: How to calculate the transition rates.
                Can be "exact" or "approximation".
                Defaults to "exact".

        Returns:
            The relevant states and the transition rates.

        """
        return self._get_transition_rates("spontaneous", unit=unit, method=method)

    @overload
    def get_black_body_transition_rates(
        self,
        temperature: Union[float, "PintFloat"],
        temperature_unit: Optional[str] = None,
        *,
        method: TransitionRateMethod = "exact",
    ) -> tuple[list["Self"], "PintArray"]: ...

    @overload
    def get_black_body_transition_rates(
        self,
        temperature: "PintFloat",
        *,
        unit: str,
        method: TransitionRateMethod = "exact",
    ) -> tuple[list["Self"], "NDArray"]: ...

    @overload
    def get_black_body_transition_rates(
        self,
        temperature: float,
        temperature_unit: str,
        unit: str,
        method: TransitionRateMethod = "exact",
    ) -> tuple[list["Self"], "NDArray"]: ...

    def get_black_body_transition_rates(
        self,
        temperature: Union[float, "PintFloat"],
        temperature_unit: Optional[str] = None,
        unit: Optional[str] = None,
        method: TransitionRateMethod = "exact",
    ) -> tuple[list["Self"], Union["PintArray", "NDArray"]]:
        """Calculate the black body transition rates for the Rydberg state.

        The black body transitions rates are given by the Einstein B coefficients,
        with a weight factor given by Planck's law.

        Args:
            temperature: The temperature, for which to calculate the black body transition rates.
            temperature_unit: The unit of the temperature.
                Default None will assume the temperature is given as pint quantity.
            unit: The unit to which to convert the result.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.
            method: How to calculate the transition rates.
                Can be "exact" or "approximation".
                Defaults to "exact".

        Returns:
            The relevant states and the transition rates.

        """
        if temperature_unit is not None:
            temperature = ureg.Quantity(temperature, temperature_unit)
        temperature_au = (temperature * ureg.Quantity(1, "boltzmann_constant")).to_base_units().magnitude
        return self._get_transition_rates("black_body", temperature_au, unit=unit, method=method)

    def _get_transition_rates(
        self,
        which_transitions: Literal["spontaneous", "black_body"],
        temperature_au: Union[float, None] = None,
        unit: Optional[str] = None,
        method: TransitionRateMethod = "exact",
    ) -> tuple[list["Self"], Union["PintArray", "NDArray"]]:
        assert which_transitions in ["spontaneous", "black_body"]

        is_spontaneous = which_transitions == "spontaneous"
        n_max = self.n + 30

        transition_rates_au: NDArray
        if method == "exact":
            # see https://en.wikipedia.org/wiki/Einstein_coefficients
            relevant_states, energy_differences, electric_dipole_moments = self._get_list_of_dipole_coupled_states(
                1, n_max, only_smaller_energy=is_spontaneous
            )
            transition_rates_au = np.abs(electric_dipole_moments) ** 2
        elif method == "approximation":
            # see https://journals.aps.org/pra/pdf/10.1103/PhysRevA.79.052504
            relevant_states, energy_differences, radial_matrix_elements = (
                self._get_list_of_radial_dipole_coupled_states(1, n_max, only_smaller_energy=is_spontaneous)
            )
            l_list = np.array([state.l for state in relevant_states])
            lmax_list = np.array([max(self.l, l) for l in l_list])
            transition_rates_au = np.abs(radial_matrix_elements) ** 2 * lmax_list / (2 * l_list + 1)
        else:
            raise ValueError(f"Method {method} not supported.")

        transition_rates_au *= (
            (4 / 3) * energy_differences**2 / ureg.Quantity(1, "speed_of_light").to_base_units().magnitude ** 3
        )

        if is_spontaneous:
            transition_rates_au *= energy_differences
        else:
            assert temperature_au is not None, "Temperature must be given for black body transitions."
            # for numerical stability we use 1 / exprel(x) = x / (exp(x) - 1)
            if temperature_au == 0:
                transition_rates_au *= 0
            else:
                transition_rates_au *= temperature_au / exprel(energy_differences / temperature_au)

        if unit == "a.u.":
            # Note in a.u.: hbar = 1 and 4 pi * epsilon_0 = 1
            return relevant_states, transition_rates_au

        transition_rates = transition_rates_au / BaseQuantities["TIME"]

        if unit is None:
            return relevant_states, transition_rates
        return relevant_states, transition_rates.to(unit).magnitude

    @overload
    def get_lifetime(
        self,
        temperature: Union[float, "PintFloat", None] = None,
        temperature_unit: Optional[str] = None,
        *,
        method: TransitionRateMethod = "exact",
    ) -> "PintFloat": ...

    @overload
    def get_lifetime(
        self,
        *,
        unit: str,
        method: TransitionRateMethod = "exact",
    ) -> float: ...

    @overload
    def get_lifetime(
        self,
        temperature: "PintFloat",
        *,
        unit: str,
        method: TransitionRateMethod = "exact",
    ) -> float: ...

    @overload
    def get_lifetime(
        self,
        temperature: float,
        temperature_unit: str,
        unit: str,
        method: TransitionRateMethod = "exact",
    ) -> float: ...

    def get_lifetime(
        self,
        temperature: Union[float, "PintFloat", None] = None,
        temperature_unit: Optional[str] = None,
        unit: Optional[str] = None,
        method: TransitionRateMethod = "exact",
    ) -> Union["PintFloat", float]:
        r"""Calculate the lifetime of the Rydberg state.

        The lifetime is given by the inverse of the sum of the transition rates:

        .. math::
            \tau = \frac{1}{\\sum_i A_i}

        where :math:`A_i` are the transition rates
        (see `get_spontaneous_transition_rates` and `get_black_body_transition_rates`).

        Args:
            temperature: The temperature, for which to calculate the lifetime.
                Default None will only consider the spontaneous transition rates for the lifetime.
            temperature_unit: The unit of the temperature.
                Default None will assume the temperature is given as pint quantity.
            unit: The unit to which to convert the result to.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.
            method: How to calculate the transition rates.
                Can be "exact" or "approximation".
                Defaults to "exact".

        Returns:
            The lifetime of the Rydberg state in the given unit.

        """
        _, transition_rates = self.get_spontaneous_transition_rates(unit="a.u.", method=method)
        if temperature is not None:
            _, black_body_transition_rates = self.get_black_body_transition_rates(
                temperature,  # type: ignore [arg-type]
                temperature_unit,  # type: ignore [arg-type]
                unit="a.u.",
                method=method,
            )
            transition_rates = np.append(transition_rates, black_body_transition_rates)

        lifetime_au: float = 1 / np.sum(transition_rates)

        if unit == "a.u.":
            return lifetime_au
        lifetime: PintFloat = lifetime_au * BaseQuantities["TIME"]
        if unit is None:
            return lifetime
        return lifetime.to(unit).magnitude

    def _get_list_of_dipole_coupled_states(
        self, n_min: int, n_max: int, only_smaller_energy: bool = True
    ) -> tuple[list["Self"], "NDArray", "NDArray"]:
        if self.m is None:
            raise ValueError("m must be set to get the dipole coupled states.")

        relevant_states = []
        energy_differences = []
        electric_dipole_moments = []
        for n in range(n_min, n_max + 1):
            for l in [self.l - 1, self.l + 1]:
                for j_tot in np.arange(self.j_tot - 1, self.j_tot + 2):
                    for m in np.arange(self.m - 1, self.m + 2):
                        if (
                            not 0 <= l < n
                            or not -j_tot <= m <= j_tot
                            or not abs(l - self.s_tot) <= j_tot <= l + self.s_tot
                            or not self.element.is_allowed_shell(n, l, self.s_tot)
                        ):
                            continue
                        other = self.__class__(self.species, n=n, l=l, j_tot=float(j_tot), m=float(m))
                        assert other.m is not None
                        if other.get_energy("a.u.") < self.get_energy("a.u.") or not only_smaller_energy:
                            relevant_states.append(other)
                            energy_differences.append(self.get_energy("a.u.") - other.get_energy("a.u."))
                            q = round(other.m - self.m)
                            dipole_moment_au = self.calc_matrix_element(other, "ELECTRIC", 1, 1, q=q, unit="a.u.")
                            electric_dipole_moments.append(dipole_moment_au)

                            assert dipole_moment_au != 0, (
                                f"Electric dipole moment between {self} and {other} is zero. This should not happen."
                            )

        return relevant_states, np.array(energy_differences), np.array(electric_dipole_moments)

    def _get_list_of_radial_dipole_coupled_states(
        self, n_min: int, n_max: int, only_smaller_energy: bool = True
    ) -> tuple[list["Self"], "NDArray", "NDArray"]:
        relevant_states = []
        energy_differences = []
        radial_matrix_elements = []
        for n in range(n_min, n_max + 1):
            for l in [self.l - 1, self.l + 1]:
                for j_tot in np.arange(self.j_tot - 1, self.j_tot + 2):
                    if (
                        not 0 <= l < n
                        or not abs(l - self.s_tot) <= j_tot <= l + self.s_tot
                        or not self.element.is_allowed_shell(n, l, self.s_tot)
                    ):
                        continue
                    other = self.__class__(self.species, n=n, l=l, j_tot=float(j_tot))
                    if other.get_energy("a.u.") < self.get_energy("a.u.") or not only_smaller_energy:
                        relevant_states.append(other)
                        energy_differences.append(self.get_energy("a.u.") - other.get_energy("a.u."))
                        radial_me_au = calc_radial_matrix_element(self, other, 1)
                        radial_matrix_elements.append(radial_me_au)

                        assert radial_me_au != 0, (
                            f"Reduced electric dipole moment between {self} and {other} is zero. This should not happen"
                        )

        return relevant_states, np.array(energy_differences), np.array(radial_matrix_elements)


RydbergState = RydbergStateSQDT
