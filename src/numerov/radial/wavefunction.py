import logging
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from numerov.radial.numerov import _run_numerov_integration_python, run_numerov_integration

if TYPE_CHECKING:
    from numerov.model import ModelPotential
    from numerov.radial.grid import Grid


logger = logging.getLogger(__name__)


class Wavefunction:
    r"""An object containing all the relevant information about the radial wavefunction.

    Attributes:
        wlist: The dimensionless and scaled wavefunction
            w(z) = z^{-1/2} \tilde{u}(x=z^2) = (r/a_0)^{-1/4} \\sqrt(a_0) r R(r) evaluated at the zlist values.
        ulist: The corresponding dimensionless wavefunction \tilde{u}(x) = sqrt(a_0) r R(r).
        Rlist: The corresponding dimensionless radial wavefunction \tilde{R}(r) = a_0^{-3/2} R(r).

    """

    def __init__(
        self,
        grid: "Grid",
        model: "ModelPotential",
    ) -> None:
        """Create a Wavefunction object.

        Args:
            grid: The grid object containing the radial grid information.
            model: The model potential object containing the potential information.

        """
        self._grid = grid
        self._model = model

        self._wlist: np.ndarray = None
        self._ulist: np.ndarray = None
        self._Rlist: np.ndarray = None

    @property
    def grid(self) -> "Grid":
        """The grid object containing the radial grid information."""
        return self._grid

    @property
    def model(self) -> "ModelPotential":
        """The model potential object containing the potential information."""
        return self._model

    @property
    def wlist(self) -> np.ndarray:
        r"""The dimensionless scaled wavefunction w(z) = z^{-1/2} \tilde{u}(x=z^2) = (r/a_0)^{-1/4} sqrt(a_0) r R(r)."""
        if self._wlist is None:
            self.integrate()
        return self._wlist

    @cached_property
    def ulist(self) -> np.ndarray:
        r"""The dimensionless wavefunction \tilde{u}(x) = sqrt(a_0) r R(r)."""
        return np.sqrt(self.grid.zlist) * self.wlist

    @cached_property
    def Rlist(self) -> np.ndarray:
        r"""The radial wavefunction R(r) in atomic units."""
        return self.ulist / self.grid.xlist

    def integrate(self, run_backward: bool = True, w0: float = 1e-10, _use_njit: bool = True) -> None:
        r"""Run the Numerov integration of the radial Schrödinger equation.

        The resulting radial wavefunctions are then stored as attributes, where
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

        Args:
            run_backward (default: True): Wheter to integrate the radial Schrödinger equation "backward" of "forward".
            w0 (default: 1e-10): The initial magnitude of the radial wavefunction at the outer boundary.
                For forward integration we set w[0] = 0 and w[1] = w0,
                for backward integration we set w[-1] = 0 and w[-2] = (-1)^{(n - l - 1) % 2} * w0.
            _use_njit (default: True): Whether to use the fast njit version of the Numerov integration.

        """
        if self._wlist is not None:
            raise ValueError("The wavefunction was already integrated, you should not integrate it again.")

        # Note: Inside this method we use y and x like it is used in the numerov function
        # and not like in the rest of this class, i.e. y = w(z) and x = z
        grid = self.grid

        glist = 8 * self.model.ritz_params.mu * grid.zlist**2 * (self.model.energy - self.model.calc_V_tot(grid.xlist))

        if run_backward:
            # Note: n - l - 1 is the number of nodes of the radial wavefunction
            # Thus, the sign of the wavefunction at the outer boundary is (-1)^{(n - l - 1) % 2}
            y0, y1 = 0, (-1) ** ((self.model.n - self.model.l - 1) % 2) * w0
            x_start, x_stop, dx = grid.zmax, grid.zmin, -grid.dz
            g_list_directed = glist[::-1]
            # We set x_min to the classical turning point
            # after x_min is reached in the integration, the integration stops, as soon as it crosses the x-axis again
            # or it reaches a local minimum (thus goiing away from the x-axis)
            x_min = self.model.calc_z_turning_point("classical", dz=grid.dz)
            x_min = max(x_min, 5 * dx)

        else:  # forward
            y0, y1 = 0, w0
            x_start, x_stop, dx = grid.zmin, grid.zmax, grid.dz
            g_list_directed = glist
            x_min = np.sqrt(self.model.n * (self.model.n + 15))

        if _use_njit:
            wlist = run_numerov_integration(x_start, x_stop, dx, y0, y1, g_list_directed, x_min)
        else:
            logger.warning("Using python implementation of Numerov integration, this is much slower!")
            wlist = _run_numerov_integration_python(x_start, x_stop, dx, y0, y1, g_list_directed, x_min)

        if run_backward:
            wlist = np.array(wlist)[::-1]
            grid.set_grid_range(step_start=grid.steps - len(wlist))
        else:
            wlist = np.array(wlist)
            grid.set_grid_range(step_stop=len(wlist))

        # normalize the wavefunction, see docstring
        norm = np.sqrt(2 * np.sum(wlist**2 * grid.zlist**2) * grid.dz)
        wlist /= norm

        self._wlist = wlist

        self.sanity_check(x_stop, run_backward)

    def sanity_check(self, z_stop: float, run_backward: bool) -> None:
        """Do some sanity checks on the wavefunction.

        Check if the wavefuntion fulfills the following conditions:
        - The wavefunction is positive (or zero) at the inner boundary.
        - The wavefunction is close to zero at the inner boundary.
        - The wavefunction is close to zero at the outer boundary.
        - The wavefunction has exactly (n - l - 1) nodes.
        - The integration stopped before z_stop (for l>0)
        """
        grid = self.grid

        idmax = np.argmax(np.abs(self.wlist))
        if run_backward and idmax < 0.05 * grid.steps:
            logger.error(
                "The maximum of the wavefunction is close to the inner boundary (idmax=%s) "
                + "probably due to inner divergence of the wavefunction. ",
                idmax,
            )
            logger.warning("Trying to fix this, but the result migth still be incorrect or at least inprecise.")
            wmax = np.max(self.wlist[int(0.1 * grid.steps) :])
            wmin = np.min(self.wlist[int(0.1 * grid.steps) :])
            self._wlist *= (self.wlist <= wmax) * (self.wlist >= wmin)
            norm = np.sqrt(2 * np.sum(self.wlist**2 * grid.zlist**2) * grid.dz)
            self._wlist /= norm

        if self.wlist[0] < 0:
            logger.warning("The wavefunction is negative at the inner boundary, %s", self.wlist[0])

        max_ind = {0: 1, 1: 1, 2: 1, 3: 4, 4: 6, 5: 8}.get(self.model.l, 10)
        if self.model.l != 0 and not 0 < np.mean(self.wlist[:max_ind]) < 1e-3:
            logger.warning(
                "The wavefunction (for l=%d) is not close to zero (or negative) at the inner boundary, mean=%.2e",
                self.model.l,
                np.mean(self.wlist[:max_ind]),
            )

        outer_wf = self.wlist[int(0.95 * grid.steps) :]
        if np.mean(outer_wf) > 1e-7:
            logger.warning(
                "The wavefunction is not close to zero at the outer boundary, mean=%.2e, outer_wf=%s",
                np.mean(outer_wf),
                outer_wf,
            )

        # TODO instead of just checking the mean of the wf also check the percantage of the integral
        # i.e. something like \int_{r_outer_bound}^{\inf} r^2 |R(r)|^2 dr < tol
        # and \int_{0}^{r_inner_bound} r^2 |R(r)|^2 dr < tol

        # Check the number of nodes
        nodes = np.sum(np.abs(np.diff(np.sign(self.wlist)))) // 2
        if nodes != self.model.n - self.model.l - 1:
            logger.warning(
                f"The wavefunction has {nodes} nodes, but should have {self.model.n - self.model.l - 1} nodes."
            )

        # Check that numerov stopped and did not run until x_stop
        if self.model.l > 0:
            if run_backward and z_stop > grid.zlist[0] - grid.dz / 2:
                logger.warning("The integration did not stop at the zmin boundary, z=%s, %s", grid.zlist[0], z_stop)
            if not run_backward and z_stop < grid.zlist[-1] + grid.dz / 2:
                logger.warning("The integration did not stop at the zmax boundary, z=%s", grid.zlist[-1])
        elif self.model.l == 0 and run_backward:
            if z_stop > 1.5 * grid.dz:
                logger.warning("The integration for l=0 should go until z=dz, but a z_stop=%s was used.", z_stop)
            elif grid.zlist[0] > 2.5 * grid.dz:
                # zlist[0] should be dz, but if it is 2 * dz this is also fine
                # e.g. this might happen if the integration just stopped at the last step due to a negative y value
                logger.warning(
                    "The integration for l=0 did stop before the zmin boundary, z=%s, %s", grid.zlist[0], grid.dz
                )
