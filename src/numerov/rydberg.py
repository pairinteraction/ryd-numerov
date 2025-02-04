import logging
from dataclasses import dataclass
from typing import Optional, TypeVar, Union

import numpy as np

from numerov.model_potential import ModelPotential
from numerov.radial.grid import Grid
from numerov.radial.numerov import _run_numerov_integration_python, run_numerov_integration

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
        w0 (default: 1e-10): The initial magnitude of the radial wavefunction at the outer boundary.
            For forward integration we set w[0] = 0 and w[1] = w0,
            for backward integration we set w[-1] = 0 and w[-2] = (-1)^{(n - l - 1) % 2} * w0.

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
    w0: float = 1e-10
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
        assert not hasattr(self, "_grid"), "The grid object was already created."

        if self.run_backward:
            if xmin is None:
                # we set xmin explicitly to small,
                # since the integration will automatically stop after the turning point,
                # and as soon as the wavefunction is close to zero
                if self.l <= 10:
                    xmin = 0
                else:
                    z_i = self.model.calc_z_turning_point("hydrogen")
                    xmin = max(0, 0.5 * z_i**2 - 25)
            if xmax is None:
                xmax = 2 * self.n * (self.n + 25)

        else:  # forward integration
            if xmin is None:
                xmin = 0
            if xmax is None:
                xmax = 2 * self.n * (self.n + 25)

        # Since the potential diverges at z=0 we set the minimum zmin to 2 * dz
        zmin = max(np.sqrt(xmin), 2 * dz)
        zmax = np.sqrt(xmax)

        # put all grid points on a standard grid, i.e. 0, dz, 2dz, ...
        # this is necessary to allow integration of two different wavefunctions
        zmin = (zmin // dz) * dz

        # set the grid object
        self._grid = Grid(zmin, zmax, dz)

    def integrate(self) -> None:
        r"""Run the Numerov integration of the radial Schrödinger equation.

        The resulting raidal wavefunctions are then stored as attributes, where
        - wlist is the dimensionless and scaled wavefunction w(z)
        - ulist is the dimensionless wavefunction \tilde{u}(x)
        - Rlist is the radial wavefunction R(r) in atomic units

        The radial wavefunction are related as follows:

        .. math::
            \tilde{u}(x) = \sqrt(a_0) r R(r)

        .. math::
            w(z) = z^{-1/2} \tilde{u}(x=z^2) = (r/a_0)^{-1/4} \sqrt(a_0) r R(r)


        where z = sqrt(r/a_0) is the dimensionless scaled coordinate.

        The resulting radial wavefunction is normalized such that

        .. math::
            \int_{0}^{\infty} r^2 |R(x)|^2 dr
            = \int_{0}^{\infty} |\tilde{u}(x)|^2 dx
            = \int_{0}^{\infty} 2 z^2 |w(z)|^2 dz
            = 1
        """
        # Note: Inside this method we use y and x like it is used in the numerov function
        # and not like in the rest of this class, i.e. y = w(z) and x = z
        grid = self.grid

        glist = 8 * self.model.ritz_params.mu * grid.zlist**2 * (self.model.energy - self.model.calc_V_tot(grid.xlist))

        if self.run_backward:
            # Note: n - l - 1 is the number of nodes of the radial wavefunction
            # Thus, the sign of the wavefunction at the outer boundary is (-1)^{(n - l - 1) % 2}
            y0, y1 = 0, (-1) ** ((self.n - self.l - 1) % 2) * self.w0
            x_start, x_stop, dx = grid.zmax, grid.zmin, -grid.dz
            g_list_directed = glist[::-1]
            # We set x_min to the classical turning point
            # after x_min is reached in the integration, the integration stops, as soon as it crosses the x-axis again
            # or it reaches a local minimum (thus goiing away from the x-axis)
            x_min = self.model.calc_z_turning_point("classical")

        else:  # forward
            y0, y1 = 0, self.w0
            x_start, x_stop, dx = grid.zmin, grid.zmax, grid.dz
            g_list_directed = glist
            x_min = np.sqrt(self.n * (self.n + 15))

        if self._use_njit:
            wlist = run_numerov_integration(x_start, x_stop, dx, y0, y1, g_list_directed, x_min)
        else:
            logger.warning("Using python implementation of Numerov integration, this is much slower!")
            wlist = _run_numerov_integration_python(x_start, x_stop, dx, y0, y1, g_list_directed, x_min)

        if self.run_backward:
            self.wlist = np.array(wlist)[::-1]
            grid.set_grid_range(step_start=grid.steps - len(self.wlist))
        else:
            self.wlist = np.array(wlist)
            grid.set_grid_range(step_stop=len(self.wlist))

        # normalize the wavefunction, see docstring
        norm = np.sqrt(2 * np.sum(self.wlist**2 * grid.zlist**2) * grid.dz)
        self.wlist /= norm

        self.ulist = np.sqrt(grid.zlist) * self.wlist
        self.Rlist = self.ulist / grid.xlist

        self.sanity_check_wavefunction()

    def sanity_check_wavefunction(self) -> None:
        """Do some sanity checks on the wavefunction.

        Check if the wavefuntion fulfills the following conditions:
        - The wavefunction is positive (or zero) at the inner boundary.
        - The wavefunction is close to zero at the inner boundary.
        - The wavefunction is close to zero at the outer boundary.
        - The wavefunction has exactly (n - l - 1) nodes.
        """
        grid = self.grid

        outer_wf = self.wlist[int(0.95 * grid.steps) :]
        inner_wf = self.wlist[:10]

        if inner_wf[0] < 0 or np.mean(inner_wf) < 0:
            logger.warning("The wavefunction is (mostly) negative at the inner boundary, %s", inner_wf)

        if np.mean(inner_wf) > 1e-4:
            logger.warning(
                "The wavefunction is not close to zero at the inner boundary, mean=%.2e, inner_wf=%s",
                np.mean(inner_wf),
                inner_wf,
            )

        if np.mean(outer_wf) > 1e-7:
            logger.warning(
                "The wavefunction is not close to zero at the outer boundary, mean=%.2e, outer_wf=%s",
                np.mean(outer_wf),
                outer_wf,
            )

        # Check the number of nodes
        nodes = np.sum(np.abs(np.diff(np.sign(self.wlist)))) // 2
        if nodes != self.n - self.l - 1:
            logger.warning(f"The wavefunction has {nodes} nodes, but should have {self.n - self.l - 1} nodes.")

        # TODO instead of just checking the mean of the wf also check the percantage of the integral
        # i.e. something like \int_{r_outer_bound}^{\inf} r^2 |R(r)|^2 dr < tol
        # and \int_{0}^{r_inner_bound} r^2 |R(r)|^2 dr < tol

        # TODO check that numerov stopped and did not run until x_stop
