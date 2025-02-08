import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union, overload

import numpy as np

from numerov.angular.angular_matrix_element import OperatorType, calc_angular_matrix_element
from numerov.model.model_potential import ModelPotential
from numerov.radial.grid import Grid
from numerov.radial.radial_matrix_element import calc_radial_matrix_element
from numerov.radial.wavefunction import Wavefunction
from numerov.units import BaseQuantities

if TYPE_CHECKING:
    from pint.facets.plain import PlainQuantity
    from typing_extensions import Self


logger = logging.getLogger(__name__)


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
    m: Union[int, float] = None
    s: Union[int, float] = None

    def __post_init__(self) -> None:
        if self.s is None:
            if self.species.endswith("singlet"):
                self.s = 0
            elif self.species.endswith("triplet"):
                self.s = 1
            else:
                self.s = 0.5

        assert isinstance(self.s, (float, int)), "s must be a float or int"
        assert self.n >= 1, "n must be larger than 0"
        assert 0 <= self.l <= self.n - 1, "l must be between 0 and n - 1"
        assert self.j >= abs(self.l - self.s) and self.j <= self.l + self.s, "j must be between l - s and l + s"
        assert (self.j + self.s) % 1 == 0, "j and s both must be integer or half-integer"

        self._model: ModelPotential = None
        self._grid: Grid = None
        self._wavefunction: Wavefunction = None

    @property
    def model(self) -> ModelPotential:
        if self._model is None:
            self.create_model()
        return self._model

    def create_model(self, qdd_path: Optional[str] = None, add_spin_orbit: bool = True) -> None:
        """Create the model potential for the Rydberg state.

        Args:
            qdd_path: Optional path to a SQLite database file containing the quantum defects.
            Default None, i.e. use the default quantum_defects.sql.
            add_spin_orbit: Whether to include the spin-orbit interaction in the model potential.
            Defaults to True.

        """
        if self._model is not None:
            raise ValueError("The model was already created, you should not create a different model.")
        self._model = ModelPotential(self.species, self.n, self.l, self.s, self.j, qdd_path, add_spin_orbit)

    @property
    def energy(self) -> float:
        """The energy of the Rydberg state in atomic units."""
        return self.model.energy

    @property
    def grid(self) -> Grid:
        if self._grid is None:
            self.create_grid()
        return self._grid

    def create_grid(
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        dz: float = 1e-2,
    ) -> None:
        """Create the grid object for the integration of the radial Schrödinger equation.

        Args:
            xmin (default TODO): The minimum value of the radial coordinate
            in dimensionless units (x = r/a_0).
            xmax (default TODO): The maximum value of the radial coordinate
            in dimensionless units (x = r/a_0).
            dz (default 1e-2): The step size of the integration (z = r/a_0).

        """
        if self._grid is not None:
            raise ValueError("The grid was already created, you should not create a different grid.")

        if xmin is None:
            # we set xmin explicitly to small,
            # since the integration will automatically stop after the turning point,
            # and as soon as the wavefunction is close to zero
            if self.l <= 10:
                xmin = 0
            else:
                z_i = self.model.calc_z_turning_point("hydrogen", dz=1e-2)
                xmin = max(0, 0.5 * z_i**2 - 25)
        if xmax is None:
            # This is an empirical formula for the maximum value of the radial coordinate
            # it takes into account that for large n but small l the wavefunction is very extended
            xmax = 2 * self.n * (self.n + 15 + (self.n - self.l) / 4)

        # Since the potential diverges at z=0 we set the minimum zmin to 2 * dz
        zmin = max(np.sqrt(xmin), dz)
        zmax = np.sqrt(xmax)

        # put all grid points on a standard grid, i.e. 0, dz, 2dz, ...
        # this is necessary to allow integration of two different wavefunctions
        zmin = (zmin // dz) * dz

        # set the grid object
        self._grid = Grid(zmin, zmax, dz)

    @property
    def wavefunction(self) -> Wavefunction:
        if self._wavefunction is None:
            self.integrate_wavefunction()
        return self._wavefunction

    def integrate_wavefunction(self, run_backward: bool = True, w0: float = 1e-10, _use_njit: bool = True) -> None:
        if self._wavefunction is not None:
            raise ValueError("The wavefunction was already integrated, you should not integrate it again.")
        self._wavefunction = Wavefunction(self.grid, self.model)
        self._wavefunction.integrate(run_backward, w0, _use_njit)

    @overload
    def calc_radial_matrix_element(self, other: "Self", k_radial: int) -> "PlainQuantity[float]": ...

    @overload
    def calc_radial_matrix_element(self, other: "Self", k_radial: int, unit: str) -> float: ...

    def calc_radial_matrix_element(self, other: "Self", k_radial: int, unit: Optional[str] = None):
        radial_matrix_element_au = calc_radial_matrix_element(self, other, k_radial)
        if unit == "a.u.":
            return radial_matrix_element_au
        radial_matrix_element = radial_matrix_element_au * BaseQuantities["RADIAL_MATRIX_ELEMENT"]
        if unit is None:
            return radial_matrix_element
        return radial_matrix_element.to(unit).magnitude

    @overload
    def calc_angular_matrix_element(
        self, other: "Self", operator: OperatorType, k_angular: int, q: int
    ) -> "PlainQuantity[float]": ...

    @overload
    def calc_angular_matrix_element(
        self, other: "Self", operator: OperatorType, k_angular: int, q: int, unit: str
    ) -> float: ...

    def calc_angular_matrix_element(
        self, other: "Self", operator: OperatorType, k_angular: int, q: int, unit: Optional[str] = None
    ):
        angular_matrix_element_au = calc_angular_matrix_element(self, other, operator, k_angular, q)
        if unit == "a.u.":
            return angular_matrix_element_au
        angular_matrix_element = angular_matrix_element_au * BaseQuantities["RADIAL_MATRIX_ELEMENT"]
        if unit is None:
            return angular_matrix_element
        return angular_matrix_element.to(unit).magnitude

    @overload
    def calc_multipole_matrix_element(
        self, other: "Self", k_radial: int, k_angular: int, q: int
    ) -> "PlainQuantity[float]": ...

    @overload
    def calc_multipole_matrix_element(
        self, other: "Self", k_radial: int, k_angular: int, q: int, unit: str
    ) -> float: ...

    def calc_multipole_matrix_element(
        self, other: "Self", k_radial: int, k_angular: int, q: int, unit: Optional[str] = None
    ):
        r"""Calculate the multipole matrix element.

        Calculate the multipole matrix element between two Rydberg states
        \ket{self}=\ket{n',l',j',m'} and \ket{other}= \ket{n,l,j,m}.

        .. math::
            \langle n,l,j,m,s | r^k_radial p_{k_angular,q} | n',l',j',m',s' \rangle

        where p_{k_angular,q} is the spherical multipole operators of rank k_angular and component q.

        Args:
            other: The other Rydberg state \ket{n,l,j,m,s} to which to calculate the matrix element.
            k_radial: The radial matrix element power k.
            k_angular: The rank of the angular operator.
            q: The component of the angular operator.
            unit: The unit to which to convert the radial matrix element.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.

        Returns:
            The multipole matrix element.

        """
        operator = "p"
        radial_matrix_element_au = calc_radial_matrix_element(self, other, k_radial)
        angular_matrix_element_au = calc_angular_matrix_element(self, other, operator, k_angular, q)
        multipole_matrix_element_au = radial_matrix_element_au * angular_matrix_element_au
        if unit == "a.u.":
            return multipole_matrix_element_au
        multipole_matrix_element = (
            multipole_matrix_element_au * BaseQuantities["RADIAL_MATRIX_ELEMENT"] ** k_radial * BaseQuantities["CHARGE"]
        )
        if unit is None:
            return multipole_matrix_element
        return multipole_matrix_element.to(unit).magnitude
