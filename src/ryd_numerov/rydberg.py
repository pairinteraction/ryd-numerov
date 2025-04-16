import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Union, get_args, overload

import numpy as np
from scipy.special import exprel

from ryd_numerov.angular import calc_angular_matrix_element
from ryd_numerov.model import Model
from ryd_numerov.radial import Grid, Wavefunction, calc_radial_matrix_element
from ryd_numerov.units import BaseQuantities, OperatorType, ureg

if TYPE_CHECKING:
    from typing_extensions import Self

    from ryd_numerov.model import Database
    from ryd_numerov.units import NDArray, PintArray, PintFloat


logger = logging.getLogger(__name__)

TransitionRateMethod = Literal["exact", "approximation"]

ALKALI_SPECIES = ["H", "Li", "Na", "K", "Rb", "Cs", "Fr"]
ALKALINE_EARTH_SPECIES = ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"]


@dataclass
class RydbergState:
    r"""Create a Rydberg state, for which the radial Schrödinger equation is solved using the Numerov method.

    Integrate the radial Schrödinger equation for the Rydberg state using the Numerov method.

    We solve the radial dimensionless Schrödinger equation for the Rydberg state

    .. math::
        \frac{d^2}{dx^2} u(x) = - \left[ E - V_{eff}(x) \right] u(x)

    using the Numerov method, see `integration.run_numerov_integration`.

    Args:
        species: The Rydberg atom species for which to solve the radial Schrödinger equation.
        n: The principal quantum number of the desired electronic state.
        l: The angular momentum quantum number of the desired electronic state.
        j: The total angular momentum quantum number of the desired electronic state.
        m: The magnetic quantum number of the desired electronic state.
            Optional, only needed for concrete angular matrix elements.
        s: The spin quantum number of the desired electronic state. Default tries to infer from the species.

    """

    species: str
    n: int
    l: int
    j: Union[int, float]
    m: Union[int, float, None] = None
    s: Union[int, float] = None  # type: ignore [assignment]  # will always be set to float or int in __post_init__

    def __post_init__(self) -> None:
        if self.s is None:
            self.s = get_spin_from_species(self.species)

        assert isinstance(self.s, (float, int)), "s must be a float or int"
        assert self.n >= 1, "n must be larger than 0"
        assert 0 <= self.l <= self.n - 1, "l must be between 0 and n - 1"
        assert abs(self.l - self.s) <= self.j <= self.l + self.s, "j must be between l - s and l + s"
        assert (self.j + self.s) % 1 == 0, "j and s both must be integer or half-integer"

        self._model: Optional[Model] = None
        self._grid: Optional[Grid] = None
        self._wavefunction: Optional[Wavefunction] = None

    @property
    def is_alkali(self) -> bool:
        return self.species.split("_")[0] in ALKALI_SPECIES

    @property
    def is_alkaline_earth(self) -> bool:
        return self.species.split("_")[0] in ALKALINE_EARTH_SPECIES

    @property
    def model(self) -> Model:
        if self._model is None:
            return self.create_model()
        return self._model

    def create_model(self, add_spin_orbit: bool = True, database: Optional["Database"] = None) -> Model:
        """Create the model potential for the Rydberg state.

        Args:
            add_spin_orbit: Whether to include the spin-orbit interaction in the model potential.
                Defaults to True.
            database: Database object containing the quantum defects.
                Default None, i.e. use the global database instance.

        """
        if self._model is not None:
            raise ValueError("The model was already created, you should not create a different model.")

        self._model = Model(
            self.species,
            self.n,
            self.l,
            self.s,
            self.j,
            add_spin_orbit=add_spin_orbit,
            database=database,
        )
        return self._model

    @property
    def energy(self) -> float:
        """The energy of the Rydberg state in atomic units."""
        return self.model.ritz_params.get_energy(self.n)

    @property
    def grid(self) -> Grid:
        if self._grid is None:
            return self.create_grid()
        return self._grid

    def create_grid(
        self,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        dz: float = 1e-2,
    ) -> Grid:
        """Create the grid object for the integration of the radial Schrödinger equation.

        Args:
            x_min (default TODO): The minimum value of the radial coordinate
            in dimensionless units (x = r/a_0).
            x_max (default TODO): The maximum value of the radial coordinate
            in dimensionless units (x = r/a_0).
            dz (default 1e-2): The step size of the integration (z = r/a_0).

        """
        if self._grid is not None:
            raise ValueError("The grid was already created, you should not create a different grid.")

        if x_min is None:
            # we set x_min explicitly to small,
            # since the integration will automatically stop after the turning point,
            # and as soon as the wavefunction is close to zero
            if self.l <= 10:
                x_min = 0
            else:
                z_i = self.model.calc_z_turning_point("hydrogen", dz=1e-2)
                x_min = max(0, 0.5 * z_i**2 - 25)
        if x_max is None:
            # This is an empirical formula for the maximum value of the radial coordinate
            # it takes into account that for large n but small l the wavefunction is very extended
            x_max = 2 * self.n * (self.n + 15 + (self.n - self.l) / 4)

        z_min = np.sqrt(x_min)
        z_max = np.sqrt(x_max)

        # put all grid points on a standard grid, i.e. [dz, 2*dz, 3*dz, ...]
        # this is necessary to allow integration of two different wavefunctions
        z_min = (z_min // dz) * dz

        # Since the potential diverges at z=0 we set the minimum z_min to dz
        z_min = max(z_min, dz)

        # set the grid object
        self._grid = Grid(z_min, z_max, dz)
        return self._grid

    @property
    def wavefunction(self) -> Wavefunction:
        if self._wavefunction is None:
            return self.integrate_wavefunction()
        return self._wavefunction

    def integrate_wavefunction(
        self, run_backward: bool = True, w0: float = 1e-10, _use_njit: bool = True
    ) -> Wavefunction:
        if self._wavefunction is not None:
            raise ValueError("The wavefunction was already integrated, you should not integrate it again.")
        self._wavefunction = Wavefunction(self.grid, self.model)
        self._wavefunction.integrate(run_backward, w0, _use_njit)
        return self._wavefunction

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
        return radial_matrix_element.to(unit).magnitude  # type: ignore [no-any-return]  # pint typing .to(unit)

    def calc_angular_matrix_element(self, other: "Self", operator: "OperatorType", k_angular: int, q: int) -> float:
        """Calculate the dimensionless angular matrix element."""
        if self.m is None or other.m is None:
            raise ValueError("m must be set to calculate the angular matrix element.")

        return calc_angular_matrix_element(
            self.s, self.l, self.j, self.m, other.s, other.l, other.j, other.m, operator, k_angular, q
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
        \ket{self}=\ket{n',l',j',m'} and \ket{other}= \ket{n,l,j,m}.

        .. math::
            \langle n,l,j,m,s | r^k_radial \hat{O}_{k_angular,q} | n',l',j',m',s' \rangle

        where \hat{O}_{k_angular,q} is the operators of rank k_angular and component q,
        for which to calculate the matrix element.

        Args:
            other: The other Rydberg state \ket{n,l,j,m,s} to which to calculate the matrix element.
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
            # as the same dimensionality as the Bohr magneton (mu = - mu_B (g_l l + g_s s))
            # such that - mu * B (where the magnetic field B is given in dimension Tesla) is an energy
            matrix_element *= ureg.Quantity(2, "bohr_magneton")

        if unit is None:
            return matrix_element
        return matrix_element.to(unit).magnitude  # type: ignore [no-any-return]  # pint typing .to(unit)

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
                for j in np.arange(self.j - 1, self.j + 2):
                    for m in np.arange(self.m - 1, self.m + 2):
                        if (
                            not 0 <= l < n
                            or not -j <= m <= j
                            or not abs(l - self.s) <= j <= l + self.s
                            or not self.model.ground_state.is_allowed_shell(n, l)
                        ):
                            continue
                        other = self.__class__(self.species, n, l, float(j), m=float(m), s=self.s)
                        assert other.m is not None
                        other.create_model(database=self.model.database)
                        if other.energy < self.energy or not only_smaller_energy:
                            relevant_states.append(other)
                            energy_differences.append(self.energy - other.energy)
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
                for j in np.arange(self.j - 1, self.j + 2):
                    if (
                        not 0 <= l < n
                        or not abs(l - self.s) <= j <= l + self.s
                        or not self.model.ground_state.is_allowed_shell(n, l)
                    ):
                        continue
                    other = self.__class__(self.species, n, l, float(j), s=self.s)
                    other.create_model(database=self.model.database)
                    if other.energy < self.energy or not only_smaller_energy:
                        relevant_states.append(other)
                        energy_differences.append(self.energy - other.energy)
                        radial_me_au = calc_radial_matrix_element(self, other, 1)
                        radial_matrix_elements.append(radial_me_au)

                        assert radial_me_au != 0, (
                            f"Reduced electric dipole moment between {self} and {other} is zero. This should not happen"
                        )

        return relevant_states, np.array(energy_differences), np.array(radial_matrix_elements)

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
        return lifetime.to(unit).magnitude  # type: ignore [no-any-return]  # pint typing .to(unit)


def get_spin_from_species(species: str) -> float:
    if species.endswith("singlet"):
        return 0
    if species.endswith("triplet"):
        return 1
    return 0.5
