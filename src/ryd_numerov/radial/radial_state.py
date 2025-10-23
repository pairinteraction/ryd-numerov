from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from ryd_numerov.elements import BaseElement
from ryd_numerov.radial.grid import Grid
from ryd_numerov.radial.model import Model
from ryd_numerov.radial.radial_matrix_element import calc_radial_matrix_element_from_w_z
from ryd_numerov.radial.wavefunction import WavefunctionNumerov, WavefunctionWhittaker
from ryd_numerov.units import ureg

if TYPE_CHECKING:
    from ryd_numerov.radial.model import PotentialType
    from ryd_numerov.radial.radial_matrix_element import INTEGRATION_METHODS
    from ryd_numerov.radial.wavefunction import Wavefunction, WavefunctionSignConvention
    from ryd_numerov.units import PintFloat

logger = logging.getLogger(__name__)


class RadialState:
    species: str
    n: int | None
    l_r: int

    def __init__(
        self,
        species: str,
        *,
        n: int | None = None,
        nu: float,
        l_r: int,
    ) -> None:
        r"""Initialize the radial state.

        Args:
            species: Atomic species.
            nu: Effective principal quantum number of the rydberg electron,
                which is used to calculate the energy of the state.
            n: Principal quantum number of the rydberg electron.
            l_r: Orbital angular momentum quantum number of the rydberg electron.
            energy_au: The energy of the Rydberg state in atomic units ("hartree").
                Either `nu` or `energy_au` must be provided.

        """
        self.species = species

        self.n = n
        if n is not None and nu > n and abs(nu - n) < 1e-10:
            nu = n  # avoid numerical issues
        self.nu = nu
        self.l_r = l_r

        # sanity checks
        if not nu > 0:
            raise ValueError(f"nu must be larger than 0, but is {nu=}")

        if n is not None and not (isinstance(n, (int, np.integer)) and n >= 1 and n >= nu):
            raise ValueError(f"n must be an integer larger than 0 and larger (or equal) than nu, but is {n=}, {nu=}")

        if not (isinstance(l_r, (int, np.integer)) and l_r >= 0 and (n is None or l_r <= n - 1)):
            raise ValueError(f"l_r must be an integer, and between 0 and n - 1, but is {l_r=}, {n=}")

    def __repr__(self) -> str:
        species, nu, n, l_r = self.species, self.nu, self.n, self.l_r
        return f"{self.__class__.__name__}({species}, {n=}, {nu=}, {l_r=})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def model(self) -> Model:
        if not hasattr(self, "_model"):
            self.create_model()
        return self._model

    def create_model(self, potential_type: PotentialType | None = None) -> None:
        """Create the model for the Rydberg state.

        Args:
            potential_type: Which potential to use for the model.

        """
        if hasattr(self, "_model"):
            raise RuntimeError("The model was already created, you should not create it again.")

        self._model = Model(self.species, self.l_r, potential_type)

    @property
    def grid(self) -> Grid:
        """The grid object for the integration of the radial Schrödinger equation."""
        if not hasattr(self, "_grid"):
            self.create_grid()
        return self._grid

    def create_grid(
        self,
        x_min: float | None = None,
        x_max: float | None = None,
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
            if self.l_r <= 10:
                z_min = 0.0
            else:
                element = BaseElement.from_species(self.species)
                energy_au = element.calc_energy_from_nu(self.nu)
                z_min = self.model.calc_turning_point_z(energy_au)
                z_min = math.sqrt(0.5) * z_min - 3  # see also compare_z_min_cutoff.ipynb
        else:
            z_min = math.sqrt(x_min)
        # Since the potential diverges at z=0 we set the minimum z_min to dz
        z_min = max(z_min, dz)

        if x_max is None:
            n = self.n if self.n is not None else self.nu + 5
            # This is an empirical formula for the maximum value of the radial coordinate
            # it takes into account that for large n but small l the wavefunction is very extended
            x_max = 2 * n * (n + 15 + (n - self.l_r) / 4)
        z_max = math.sqrt(x_max)

        self._grid = Grid(z_min, z_max, dz)

    @property
    def wavefunction(self) -> Wavefunction:
        if not hasattr(self, "_wavefunction"):
            self._wavefunction: Wavefunction
            self.create_wavefunction()
        return self._wavefunction

    @overload
    def create_wavefunction(
        self, *, sign_convention: WavefunctionSignConvention = "positive_at_outer_bound"
    ) -> None: ...

    @overload
    def create_wavefunction(
        self,
        method: Literal["numerov"],
        sign_convention: WavefunctionSignConvention = "positive_at_outer_bound",
        *,
        run_backward: bool = True,
        w0: float = 1e-10,
        _use_njit: bool = True,
    ) -> None: ...

    @overload
    def create_wavefunction(
        self, method: Literal["whittaker"], sign_convention: WavefunctionSignConvention = "positive_at_outer_bound"
    ) -> None: ...

    def create_wavefunction(
        self,
        method: Literal["numerov", "whittaker"] = "numerov",
        sign_convention: WavefunctionSignConvention = "positive_at_outer_bound",
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

        self._wavefunction.apply_sign_convention(sign_convention)
        self._grid = self._wavefunction.grid

    @overload
    def calc_matrix_element(self, other: RadialState, k_radial: int) -> PintFloat: ...

    @overload
    def calc_matrix_element(self, other: RadialState, k_radial: int, unit: str) -> float: ...

    def calc_matrix_element(
        self,
        other: RadialState,
        k_radial: int,
        unit: str | None = None,
        *,
        integration_method: INTEGRATION_METHODS = "sum",
    ) -> PintFloat | float:
        r"""Calculate the radial matrix element <self | r^k_radial | other>.

        Computes the integral

        .. math::
            \int_{0}^{\infty} dr r^2 r^k_{radial} R_1(r) R_2(r)
            = a_0^k_{radial} \int_{0}^{\infty} dx x^k_{radial} \tilde{u}_1(x) \tilde{u}_2(x)
            = a_0^k_{radial} \int_{0}^{\infty} dz 2 z^{2 + 2k_{radial}} w_1(z) w_2(z)

        where R_1 and R_2 are the radial wavefunctions of self and other,
        and w(z) = z^{-1/2} \tilde{u}(z^2) = (r/_a_0)^{1/4} \sqrt{a_0} r R(r).

        Args:
            other: Other radial state
            k_radial: Power of r in the matrix element
                (default=0, this corresponds to the overlap integral \int dr r^2 R_1(r) R_2(r))
            unit: Unit of the returned matrix element, default None returns a Pint quantity.
            integration_method: Integration method to use

        Returns:
            The radial matrix element in the desired unit.

        """
        # Ensure wavefunctions are integrated before accessing the grid
        wf1, wf2 = self.wavefunction, other.wavefunction
        radial_matrix_element_au = calc_radial_matrix_element_from_w_z(
            wf1.grid.z_list, wf1.w_list, wf2.grid.z_list, wf2.w_list, k_radial, integration_method
        )

        if unit == "a.u.":
            return radial_matrix_element_au
        radial_matrix_element: PintFloat = radial_matrix_element_au * ureg.Quantity(1, "a0") ** k_radial
        if unit is None:
            return radial_matrix_element
        return radial_matrix_element.to(unit).magnitude
