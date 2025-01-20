import logging
from dataclasses import dataclass
from typing import Optional, TypeVar, Union

import numpy as np

from numerov.grid import Grid
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
        run_backward (default: True): Wheter to integrate the radial Schrödinger equation "backward" of "forward".
        epsilon_u (default: 1e-10): The initial magnitude of the radial wavefunction at the outer boundary.
            For forward integration we set u[0] = 0 and u[1] = epsilon_u,
            for backward integration we set u[-1] = 0 and u[-2] = (-1)^{(n - l - 1) % 2} * epsilon_u.

    Attributes:
        zlist: A equidistant numpy array of the z-values at which the wavefunction is evaluated (z = sqrt(r/a_0)).
        xlist: A numpy array of the corresponding x-values at which the wavefunction is evaluated (x = r/a_0).
        wlist: The dimensionless and scaled wavefunction
            w(z) = z^{-1/2} \tilde{u}(x=z^2) = (r/a_0)^{-1/4} \sqrt(a_0) r R(r) evaluated at the zlist values.
        ulist: The corresponding dimensionless wavefunction \tilde{u}(x) = sqrt(a_0) r R(r).
        Rlist: The corresponding dimensionless radial wavefunction \tilde{R}(r) = a_0^{-3/2} R(r).

    """

    species: str
    n: int
    l: int
    j: float

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

    @property
    def grid(self) -> Grid:
        if not hasattr(self, "_grid"):
            self.create_grid()
        return self._grid

    def create_grid(
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        dz: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> None:
        """Create the grid object for the integration of the radial Schrödinger equation.

        Args:
            xmin (default TODO): The minimum value of the radial coordinate
            in dimensionless units (x = r/a_0).
            xmax (default TODO): The maximum value of the radial coordinate
            in dimensionless units (x = r/a_0).
            dz (default 1e-2): The step size of the integration (z = r/a_0) (use either steps or dz).
            steps (default dz=1e-2): The number of steps of the integration (use either steps or dz).

        """
        if dz is None and steps is None:
            dz = 1e-2  # TODO 1e-2 like arc and pi fine or smaller?

        # set xmin and zmin
        if xmin is None:
            xmin = dz if dz is not None else 1e-2

            if self.l != 0 and self.run_backward:
                xmin = self.n * self.n - self.n * np.sqrt(
                    self.n * self.n - self.l * (self.l - 1)
                )  # TODO pi (l-1)*(l-1)
                xmin = 0.7 * xmin

        zmin = np.sqrt(xmin)
        if dz is not None:
            zmin = (zmin // dz) * dz  # TODO this is a hack for allowing integration of the matrix elements

        # set xmax and zmax
        if xmax is None:
            if self.run_backward:
                self.xmax = 2 * self.n * (self.n + 25)  # TODO 15 like arc and pi?
            else:
                self.xmax = 2 * self.n * (self.n + 6)
        zmax = np.sqrt(self.xmax)

        # set the grid
        self._grid = Grid(zmin, zmax, dz, steps)

    def integrate(self) -> None:
        r"""Run the Numerov integration of the radial Schrödinger equation for the desired state.

        Returns:
            zlist: A numpy array of the z-values at which the wavefunction was evaluated (z = sqrt(r/a_0))
            wlist: A numpy array of the function w(z) = z^{-1/2} \tilde{u}(x=z^2) = (r/a_0)^{-1/4} sqrt(a_0) r R(r)
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

        grid = self.grid
        glist = 8 * grid.zlist**2 * (self.model.energy - self.model.calc_V_tot(grid.xlist))
        if self._use_njit:
            self.wlist = run_numerov_integration(grid.dz, grid.steps, y0, y1, glist, self.run_backward)
        else:
            self.wlist = _python_run_numerov_integration(grid.dz, grid.steps, y0, y1, glist, self.run_backward)

        # normalize the wavefunction, see docstring
        norm = np.sqrt(2 * np.sum(self.wlist**2 * grid.zlist**2) * grid.dz)
        self.wlist /= norm

        self.ulist = np.sqrt(grid.zlist) * self.wlist
        self.Rlist = self.ulist / grid.xlist

        self.sanity_check_wavefunction()

    def sanity_check_wavefunction(self) -> None:
        # Check that xmax was chosen large enough
        grid = self.grid
        id = int(0.95 * grid.steps)
        sum_large_z = np.sqrt(2 * np.sum(self.wlist[id:] ** 2 * grid.zlist[id:] ** 2) * grid.dz)
        if sum_large_z > 1e-3:
            logger.warning(f"xmax={self.xmax} was chosen too small ({sum_large_z=}), increase xmax.")

        # Check that xmin was chosen good enough
        self.z_cutoff = 0
        id = int(0.01 * grid.steps)
        sum_small_z = np.sqrt(2 * np.sum(self.wlist[:id] ** 2 * grid.zlist[:id] ** 2) * grid.dz)
        if sum_small_z > 1e-3:
            logger.info(f"xmin={grid.xmin} was not chosen good ({sum_small_z=}), change xmin.")
            if self.wlist[0] < 0:
                logger.info(
                    "The wavefunction is negative at the inner boundary, setting all initial negative values to 0."
                )
                argmin = np.argwhere(self.wlist > 0)[0][0]
                self.z_cutoff = grid.zlist[argmin]
                self.wlist[:argmin] = 0

                # normalize the wavefunction again
                norm = np.sqrt(2 * np.sum(self.wlist**2 * grid.zlist**2) * grid.dz)
                self.wlist /= norm
            else:
                logger.warning(
                    f"xmin={grid.xmin} was not chosen good ({sum_small_z=}), "
                    "and the wavefunction is positive at the inner boundary, so we could not fix it."
                )

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
        grid = self.grid
        zlist = np.arange(grid.dz, grid.zlist[-1], grid.dz)
        V_phys = self.model.calc_V_phys(zlist**2)
        arg = np.argwhere(V_phys < self.model.energy)[0][0]
        return zlist[arg]

    def calc_z_V_eq_0(self) -> float:
        """Calculate the value of z where the effective physical potential equals zero.

        Calculate the value of z = sqrt(r / a_0) where the effective physical potential equals zero:

        .. math::
            0 = V_c(r_i) + V_l(r_i) + V_{l}(r_i) + V_{so}(r_i)

        Returns:
            z: The value of z where the effective physical potential equals zero
            in dimensionless units (z = sqrt(r/a_0)).

        """
        grid = self.grid
        zlist = np.arange(grid.dz, grid.zlist[-1], grid.dz)
        V_phys = self.model.calc_V_phys(zlist**2)
        arg = np.argwhere(V_phys < 0)[0][0]
        return zlist[arg]
