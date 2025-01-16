import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TypeVar, Union

import numpy as np

from numerov.database import QuantumDefectsDatabase
from numerov.numerov import _python_run_numerov_integration, run_numerov_integration
from numerov.units import ureg

ValueType = TypeVar("ValueType", bound=Union[float, np.ndarray])

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
        run_backward (default: True): Wheter to integrate the radial Schrödinger equation "backward" of "forward".
        epsilon_u (default: 1e-10): The initial magnitude of the radial wavefunction at the outer boundary.
            For forward integration we set u[0] = 0 and u[1] = epsilon_u,
            for backward integration we set u[-1] = 0 and u[-2] = (-1)^{(n - l - 1) % 2} * epsilon_u.
        xmin (default see `RydbergState.set_range`): The minimum value of the radial coordinate
        in dimensionless units (x = r/a_0).
        xmax (default see `RydbergState.set_range`): The maximum value of the radial coordinate
        in dimensionless units (x = r/a_0).
        dz (default see `RydbergState.set_range`): The step size of the integration (z = r/a_0).
        steps (default see `RydbergState.set_range`): The number of steps of the integration (use either steps or dx).

    Attributes:
        z_list: A equidistant numpy array of the z-values at which the wavefunction is evaluated (z = sqrt(r/a_0)).
        x_list: A numpy array of the corresponding x-values at which the wavefunction is evaluated (x = r/a_0).
        w_list: The dimensionless and scaled wavefunction
            w(z) = z^{-1/2} \tilde{u}(x=z^2) = (r/a_0)^{-1/4} \sqrt(a_0) r R(r) evaluated at the z_list values.
        u_list: The corresponding dimensionless wavefunction \tilde{u}(x) = sqrt(a_0) r R(r).
        R_list: The corresponding dimensionless radial wavefunction \tilde{R}(r) = a_0^{-3/2} R(r).

    """

    species: str
    n: int
    l: int
    j: float

    add_spin_orbit: bool = True

    run_backward: bool = True
    epsilon_u: float = 1e-10
    _use_njit: bool = True

    xmin: float = np.nan
    xmax: float = np.nan
    dz: float = np.nan
    steps: int = np.nan

    def __post_init__(self) -> None:
        self.s: Union[int, float]
        if self.species.endswith("singlet") or self.species.endswith("1"):
            self.s = 0
        elif self.species.endswith("triplet") or self.species.endswith("3"):
            self.s = 1
        else:
            self.s = 0.5

        assert self.n >= 1, "n must be larger than 0"
        assert 0 <= self.l <= self.n - 1, "l must be between 0 and n - 1"
        assert self.j >= abs(self.l - self.s) and self.j <= self.l + self.s, "j must be between l - s and l + s"
        assert (self.j + self.s) % 1 == 0, "j and s both must be integer or half-integer"

        self.load_parameters_from_database()
        self.set_range()

    def load_parameters_from_database(self) -> None:
        qdd = QuantumDefectsDatabase()

        model = qdd.get_model_potential(self.species, self.l)
        self.Z = model.Z
        self.a1, self.a2, self.a3, self.a4 = model.a1, model.a2, model.a3, model.a4
        self.ac = model.ac
        self.xc = model.rc

        ritz = qdd.get_rydberg_ritz(self.species, self.l, self.j)
        self.energy = ritz.get_energy(self.n)

    def set_range(self) -> None:
        """Automatically determine sensful default values for xmin, xmax and dz.

        The x-values represent the raidal coordinate in units of the Bohr radius a_0 (x = r / a_0).
        Furthermore, we define z-values as z = sqrt(x) = sqrt(r / a_0), which is used for the integration.
        The benefit of using z is that the nodes of the wavefunction are equally spaced in z-space,
        allowing for a computational better choice of choosing the constant step size during the integration.


        """
        if not np.isnan(self.dz) and not np.isnan(self.steps):
            raise ValueError("Use either dz or steps, not both.")
        elif np.isnan(self.dz) and np.isnan(self.steps):
            self.dz = 0.01

        if np.isnan(self.xmin):
            if not self.run_backward or self.l == 0:
                self.xmin = 1e-5
            else:  # self.run_backward
                xmin = self.n * self.n - self.n * np.sqrt(self.n * self.n - (self.l - 1) * (self.l - 1))
                xmin = xmin * 0.7  # TODO how to choose xmin?
                self.xmin = max(0.1, xmin)
        if np.isnan(self.xmax):
            if self.run_backward:
                self.xmax = 2 * self.n * (self.n + 25)
            else:
                self.xmax = 2 * self.n * (self.n + 6)
        zmin, zmax = np.sqrt(self.xmin), np.sqrt(self.xmax)
        if not np.isnan(self.dz):
            zmin = (zmin // self.dz) * self.dz  # TODO this is a hack for allowing integration of the matrix elements
            zmin = max(self.dz, zmin)  # dont allow zmin = 0

        if not np.isnan(self.steps):
            self.z_list = np.linspace(zmin, zmax, self.steps, endpoint=True)
            self.dz = self.z_list[1] - self.z_list[0]
        else:  # self.dz is not nan
            self.z_list = np.arange(zmin, zmax + self.dz, self.dz)
            self.steps = len(self.z_list)

        self.x_list = np.power(self.z_list, 2)

    @cached_property
    def V_tot(self) -> np.ndarray:
        return self.calc_V_tot(self.z_list)

    def calc_V_tot(self, z_list: np.ndarray) -> np.ndarray:
        x = z_list**2
        x2 = z_list**4
        x3 = z_list**6
        x4 = z_list**8

        Z_nl = 1 + (self.Z - 1) * np.exp(-self.a1 * x) - x * (self.a3 + self.a4 * x) * np.exp(-self.a2 * x)
        V_c = -Z_nl / x
        V_p = -self.ac / (2 * x4) * (1 - np.exp(-((x / self.xc) ** 6)))
        V_so = 0
        if self.add_spin_orbit:  # TODO or self.l > 4 ???
            alpha = ureg.Quantity(1, "fine_structure_constant").to_base_units().magnitude
            V_so = alpha / (4 * x3) * (self.j * (self.j + 1) - self.l * (self.l + 1) - self.s * (self.s + 1))
            # if x[0] < self.xc:  # TODO check if this is necessary
            #     V_so *= (x >= self.xc)
        V_l = self.l * (self.l + 1) / (2 * x2)
        V_sqrt = (3 / 32) / x2
        return V_c + V_p + V_so + V_l + V_sqrt

    @cached_property
    def g_list(self) -> np.ndarray:
        return 8 * self.z_list**2 * (self.energy - self.V_tot)

    def integrate(self) -> None:
        r"""Run the Numerov integration of the radial Schrödinger equation for the desired state.

        Returns:
            z_list: A numpy array of the z-values at which the wavefunction was evaluated (z = sqrt(r/a_0))
            w_list: A numpy array of the function w(z) = z^{-1/2} \tilde{u}(x=z^2) = (r/a_0)^{-1/4} sqrt(a_0) r R(r)
            (where we used \tilde{u}(x) = sqrt(a_0) r R(r)
                The radial wavefunction is normalized such that

                .. math::
                    \int_{0}^{\infty} r^2 |R(x)|^2 dr
                    = \int_{0}^{\infty} |\tilde{u}(x)|^2 dx = 1
                    = \int_{0}^{\infty} 2 z^2 |w(z)|^2 dz = 1

        """
        if self.run_backward:
            # Note: n - l - 1 is the number of nodes of the radial wavefunction
            # Thus, the sign of the wavefunction at the outer boundary is (-1)^{(n - l - 1) % 2}
            y0, y1 = 0, (-1) ** ((self.n - self.l - 1) % 2) * self.epsilon_u
        else:  # forward
            y0, y1 = 0, self.epsilon_u

        if self._use_njit:
            self.w_list = run_numerov_integration(self.dz, self.steps, y0, y1, self.g_list, self.run_backward)
        else:
            self.w_list = _python_run_numerov_integration(self.dz, self.steps, y0, y1, self.g_list, self.run_backward)

        # normalize the wavefunction, see docstring
        norm = np.sqrt(2 * np.sum(self.w_list**2 * self.z_list**2) * self.dz)
        self.w_list /= norm

        # Check that xmax was chosen large enough
        id = int(0.95 * self.steps)
        sum_large_z = np.sqrt(2 * np.sum(self.w_list[id:] ** 2 * self.z_list[id:] ** 2) * self.dz)
        if sum_large_z > 1e-3:
            logger.warning(f"xmax={self.xmax} was chosen too small ({sum_large_z=}), increase xmax.")

        # Check that xmin was chosen good enough
        id = int(0.01 * self.steps)
        sum_small_z = np.sqrt(2 * np.sum(self.w_list[:id] ** 2 * self.z_list[:id] ** 2) * self.dz)
        if sum_small_z > 1e-3:
            logger.info(f"xmin={self.xmin} was not chosen good ({sum_small_z=}), change xmin.")
            if self.w_list[0] < 0:
                logger.info(
                    "The wavefunction is negative at the inner boundary, setting all initial negative values to 0."
                )
                argmin = np.argwhere(self.w_list > 0)[0][0]
                self.w_list[:argmin] = 0

                # normalize the wavefunction again
                norm = np.sqrt(2 * np.sum(self.w_list**2 * self.z_list**2) * self.dz)
                self.w_list /= norm
            else:
                logger.warning(
                    f"xmin={self.xmin} was not chosen good ({sum_small_z=}), "
                    "and the wavefunction is positive at the inner boundary, so we could not fix it."
                )

        self.u_list = np.sqrt(self.z_list) * self.w_list
        self.R_list = self.u_list / self.x_list

    def calc_classical_zmin(self) -> float:
        """Calculate the classical turning point z_min.

        Calculate the classical turning point z_min, where the total energy equals the effective potential.

        Returns:
            classical_zmin: The classical turning point z_min in dimensionless units (z = sqrt(r/a_0)).

        """
        z = np.linspace(self.dz, self.z_list[-1], 10_000)
        V_tot = self.calc_V_tot(z)
        arg = np.argwhere(V_tot < self.energy)[0][0]
        self.classical_zmin = z[arg]
        return self.classical_zmin
