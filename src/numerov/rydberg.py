import logging
from dataclasses import dataclass
from typing import Optional, TypeVar, Union

import numpy as np

from numerov.model_potential import ModelPotential
from numerov.numerov import _python_run_numerov_integration, run_numerov_integration

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
        xmin (default see `RydbergState.set_range`): The minimum value of the radial coordinate
        in dimensionless units (x = r/a_0).
        xmax (default see `RydbergState.set_range`): The maximum value of the radial coordinate
        in dimensionless units (x = r/a_0).
        dz (default see `RydbergState.set_range`): The step size of the integration (z = r/a_0).
        steps (default see `RydbergState.set_range`): The number of steps of the integration (use either steps or dz).
        run_backward (default: True): Wheter to integrate the radial Schrödinger equation "backward" of "forward".
        epsilon_u (default: 1e-10): The initial magnitude of the radial wavefunction at the outer boundary.
            For forward integration we set u[0] = 0 and u[1] = epsilon_u,
            for backward integration we set u[-1] = 0 and u[-2] = (-1)^{(n - l - 1) % 2} * epsilon_u.

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

    xmin: float = np.nan
    xmax: float = np.nan
    dz: float = np.nan
    steps: int = np.nan

    run_backward: bool = True
    epsilon_u: float = 1e-10
    _use_njit: bool = True

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

        self.set_range()

    @property
    def energy(self) -> float:
        """The energy of the Rydberg state in atomic units."""
        return self.model.energy

    @property
    def model(self) -> ModelPotential:
        if not hasattr(self, "_model"):
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
        self._model = ModelPotential(self, qdd_path, add_spin_orbit)

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
                self.xmax = 2 * self.n * (self.n + 25)  # TODO 15 like arc and pi?
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

        g_list = 8 * self.z_list**2 * (self.model.energy - self.model.calc_V_tot(self.x_list))
        if self._use_njit:
            self.w_list = run_numerov_integration(self.dz, self.steps, y0, y1, g_list, self.run_backward)
        else:
            self.w_list = _python_run_numerov_integration(self.dz, self.steps, y0, y1, g_list, self.run_backward)

        # normalize the wavefunction, see docstring
        norm = np.sqrt(2 * np.sum(self.w_list**2 * self.z_list**2) * self.dz)
        self.w_list /= norm

        # Check that xmax was chosen large enough
        id = int(0.95 * self.steps)
        sum_large_z = np.sqrt(2 * np.sum(self.w_list[id:] ** 2 * self.z_list[id:] ** 2) * self.dz)
        if sum_large_z > 1e-3:
            logger.warning(f"xmax={self.xmax} was chosen too small ({sum_large_z=}), increase xmax.")

        # Check that xmin was chosen good enough
        self.z_cutoff = 0
        id = int(0.01 * self.steps)
        sum_small_z = np.sqrt(2 * np.sum(self.w_list[:id] ** 2 * self.z_list[:id] ** 2) * self.dz)
        if sum_small_z > 1e-3:
            logger.info(f"xmin={self.xmin} was not chosen good ({sum_small_z=}), change xmin.")
            if self.w_list[0] < 0:
                logger.info(
                    "The wavefunction is negative at the inner boundary, setting all initial negative values to 0."
                )
                argmin = np.argwhere(self.w_list > 0)[0][0]
                self.z_cutoff = self.z_list[argmin]
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

    @property
    def zmin(self) -> float:
        """The minimum value of the radial coordinate in dimensionless units (z = sqrt(r/a_0))."""
        return self.z_list[0]

    def calc_hydrogen_z_turning_point(self) -> float:
        r"""Calculate the hydrogen turning point z_i for the Rydberg state.

        The hydrogen turning point z_i = sqrt(r_i / a_0) is defined via the classical hydrogen turning point

        .. math::
            r_i = n^2 - n \sqrt{n^2 - l(l + 1)}

        This is the inner radius, for which in hydrogen V_c(r_i) + V_l(r_i) = E.

        Returns:
            z_i: The hydrogen turning point z_i in dimensionless units (z = sqrt(r/a_0)).

        """
        x_i = self.n**2 - self.n * np.sqrt(self.n**2 - self.l * (self.l + 1))
        return np.sqrt(x_i)

    def calc_z_turning_point(self) -> float:
        """Calculate the classical turning point z_min.

        Calculate the classical turning point z_min = sqrt(r_min / a_0),
        where the total energy equals the effective physical potential:

        .. math::
            E = V_c(r_i) + V_l(r_i) + V_{l}(r_i) + V_{so}(r_i)

        Returns:
            z_min: The classical turning point z_min in dimensionless units (z = sqrt(r/a_0)).

        """
        z_list = np.arange(self.dz, self.z_list[-1], self.dz)
        V_phys = self.model.calc_V_phys(z_list**2)
        arg = np.argwhere(V_phys < self.model.energy)[0][0]
        return z_list[arg]

    def calc_z_V_eq_0(self) -> float:
        """Calculate the value of z where the effective physical potential equals zero.

        Calculate the value of z = sqrt(r / a_0) where the effective physical potential equals zero:

        .. math::
            0 = V_c(r_i) + V_l(r_i) + V_{l}(r_i) + V_{so}(r_i)

        Returns:
            z: The value of z where the effective physical potential equals zero
            in dimensionless units (z = sqrt(r/a_0)).

        """
        z_list = np.arange(self.dz, self.z_list[-1], self.dz)
        V_phys = self.model.calc_V_phys(z_list**2)
        arg = np.argwhere(V_phys < 0)[0][0]
        return z_list[arg]
