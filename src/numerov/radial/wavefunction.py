import logging
from typing import TYPE_CHECKING

import numpy as np

from numerov.radial.numerov import _run_numerov_integration_python, run_numerov_integration

if TYPE_CHECKING:
    from numerov.model import Model
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
        model: "Model",
    ) -> None:
        """Create a Wavefunction object.

        Args:
            grid: The grid object containing the radial grid information.
            model: The model potential object containing the potential information.

        """
        self._grid = grid
        self._model = model

        self._wlist: np.ndarray = None

    @property
    def grid(self) -> "Grid":
        """The grid object containing the radial grid information."""
        return self._grid

    @property
    def model(self) -> "Model":
        """The model potential object containing the potential information."""
        return self._model

    @property
    def wlist(self) -> np.ndarray:
        r"""The dimensionless scaled wavefunction w(z) = z^{-1/2} \tilde{u}(x=z^2) = (r/a_0)^{-1/4} sqrt(a_0) r R(r)."""
        if self._wlist is None:
            self.integrate()
        return self._wlist

    @property
    def ulist(self) -> np.ndarray:
        r"""The dimensionless wavefunction \tilde{u}(x) = sqrt(a_0) r R(r)."""
        return np.sqrt(self.grid.zlist) * self.wlist

    @property
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

        glist = (
            8
            * self.model.ritz_params.mu
            * grid.zlist
            * grid.zlist
            * (self.model.energy - self.model.calc_V_tot(grid.xlist))
        )

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
            x_min = max(x_min, 5 * abs(dx), self.get_xmin())

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

        wlist = np.array(wlist)
        if run_backward:
            wlist = wlist[::-1]
            grid.set_grid_range(step_start=grid.steps - len(wlist))
        else:
            grid.set_grid_range(step_stop=len(wlist))

        # normalize the wavefunction, see docstring
        norm = np.sqrt(2 * np.sum(wlist * wlist * grid.zlist * grid.zlist) * grid.dz)
        wlist /= norm

        self._wlist = wlist

        self.sanity_check(x_stop, run_backward)

    def get_xmin(self) -> float:
        """Implement a few special cases for the xmin point of the integration."""
        species, n, l = self.model.species, self.model.n, self.model.l
        if species in ["Rb", "Cs"] and n == 4 and l == 3:
            return 2
        if species == "Sr_singlet" and n == 5 and l == 0:
            return 2

        return 0

    def sanity_check(self, z_stop: float, run_backward: bool) -> bool:
        """Do some sanity checks on the wavefunction.

        Check if the wavefuntion fulfills the following conditions:
        - The wavefunction is positive (or zero) at the inner boundary.
        - The wavefunction is close to zero at the inner boundary.
        - The wavefunction is close to zero at the outer boundary.
        - The wavefunction has exactly (n - l - 1) nodes.
        - The integration stopped before z_stop (for l>0)
        """
        grid = self.grid
        sanity_check = True
        species, n, l, j = self.model.species, self.model.n, self.model.l, self.model.j

        # Check the maximum of the wavefunction
        idmax = np.argmax(np.abs(self.wlist))
        if run_backward and idmax < 0.05 * grid.steps:
            sanity_check = False
            logger.warning(
                "The maximum of the wavefunction is close to the inner boundary (idmax=%s) "
                + "probably due to inner divergence of the wavefunction. "
                + "Trying to fix this, but the result might still be incorrect or at least inprecise.",
                idmax,
            )
            wmax = np.max(self.wlist[int(0.1 * grid.steps) :])
            wmin = np.min(self.wlist[int(0.1 * grid.steps) :])
            tol = 1e-2 * max(abs(wmax), abs(wmin))
            self._wlist *= (self.wlist <= wmax + tol) * (self.wlist >= wmin - tol)
            norm = np.sqrt(2 * np.sum(self.wlist * self.wlist * grid.zlist * grid.zlist) * grid.dz)
            self._wlist /= norm

        # Check the wavefunction at the inner boundary
        if self.wlist[0] < 0:
            sanity_check = False
            logger.warning("The wavefunction is negative at the inner boundary, %s", self.wlist[0])

        inner_ind = {0: 5, 1: 5}.get(l, 10)
        inner_weight = (
            2
            * np.sum(self.wlist[:inner_ind] * self.wlist[:inner_ind] * grid.zlist[:inner_ind] * grid.zlist[:inner_ind])
            * grid.dz
        )
        inner_weight_scaled_to_whole_grid = inner_weight * grid.steps / inner_ind

        tol = 1e-5
        if l in [4, 5, 6]:
            # apparently the wavefunction converges worse for those l values
            # maybe this has something to do with the model potential parameters, which are only given for l <= 3
            tol = 1e-4
        # for low n the wavefunction also converges bad
        if n <= 15:
            tol = 2e-4
        if n < 10:
            tol = 1e-3
        if n <= 6:
            tol = 5e-3

        # special cases of bad convergence:
        if species == "K" and l == 3:
            tol = max(tol, 5e-5)
        if (species, n, l, j) == ("Cs", 5, 2, 1.5):
            tol = max(tol, 2e-2)

        if inner_weight_scaled_to_whole_grid > tol:
            sanity_check = False
            logger.warning(
                "The wavefunction is not close to zero at the inner boundary, (inner_weight_scaled_to_whole_grid=%.2e)",
                inner_weight_scaled_to_whole_grid,
            )

        # Check the wavefunction at the outer boundary
        outer_ind = int(0.95 * grid.steps)
        outer_wf = self.wlist[outer_ind:]
        if np.mean(outer_wf) > 1e-7:
            sanity_check = False
            logger.warning(
                "The wavefunction is not close to zero at the outer boundary, mean=%.2e",
                np.mean(outer_wf),
            )

        outer_weight = 2 * np.sum(outer_wf * outer_wf * grid.zlist[outer_ind:] * grid.zlist[outer_ind:]) * grid.dz
        outer_weight_scaled_to_whole_grid = outer_weight * grid.steps / len(outer_wf)
        if outer_weight_scaled_to_whole_grid > 1e-10:
            sanity_check = False
            logger.warning(
                "The wavefunction is not close to zero at the outer boundary, (outer_weight_scaled_to_whole_grid=%.2e)",
                outer_weight_scaled_to_whole_grid,
            )

        # Check the number of nodes
        nodes = np.sum(np.abs(np.diff(np.sign(self.wlist)))) // 2
        if nodes != n - l - 1:
            sanity_check = False
            logger.warning(f"The wavefunction has {nodes} nodes, but should have {n - l - 1} nodes.")

        # Check that numerov stopped and did not run until x_stop
        if l > 0:
            if run_backward and z_stop > grid.zlist[0] - grid.dz / 2:
                sanity_check = False
                logger.warning("The integration did not stop at the zmin boundary, z=%s, %s", grid.zlist[0], z_stop)
            if not run_backward and z_stop < grid.zlist[-1] + grid.dz / 2:
                sanity_check = False
                logger.warning("The integration did not stop at the zmax boundary, z=%s", grid.zlist[-1])
        elif l == 0 and run_backward:
            if z_stop > 1.5 * grid.dz:
                sanity_check = False
                logger.warning("The integration for l=0 should go until z=dz, but a z_stop=%s was used.", z_stop)
            elif grid.zlist[0] > 2.5 * grid.dz:
                # zlist[0] should be dz, but if it is 2 * dz this is also fine
                # e.g. this might happen if the integration just stopped at the last step due to a negative y value
                sanity_check = False
                logger.warning(
                    "The integration for l=0 did stop before the zmin boundary, z=%s, %s", grid.zlist[0], grid.dz
                )

        if not sanity_check:
            logger.error(
                "The wavefunction (species=%s n=%d, l=%d, j=%.1f) has some issues.",
                self.model.species,
                n,
                l,
                j,
            )

        return sanity_check
