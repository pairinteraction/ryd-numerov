import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from ryd_numerov.radial.numerov import _run_numerov_integration_python, run_numerov_integration

if TYPE_CHECKING:
    from ryd_numerov.elements.element import Element
    from ryd_numerov.model import ModelPotential
    from ryd_numerov.radial.grid import Grid
    from ryd_numerov.units import NDArray

logger = logging.getLogger(__name__)


class Wavefunction:
    r"""An object containing all the relevant information about the radial wavefunction.

    Attributes:
        w_list: The dimensionless and scaled wavefunction
            w(z) = z^{-1/2} \tilde{u}(x=z^2) = (r/a_0)^{-1/4} \\sqrt(a_0) r R(r) evaluated at the z_list values.
        u_list: The corresponding dimensionless wavefunction \tilde{u}(x) = sqrt(a_0) r R(r).
        r_list: The corresponding dimensionless radial wavefunction \tilde{R}(r) = a_0^{-3/2} R(r).

    """

    def __init__(
        self,
        element: "Element",
        grid: "Grid",
        model_potential: "ModelPotential",
    ) -> None:
        """Create a Wavefunction object.

        Args:
            element: The element object.
            grid: The grid object.
            model_potential: The model potential object.

        """
        self.element = element
        self.grid = grid
        self.model_potential = model_potential

        self._w_list: Optional[NDArray] = None

    @property
    def w_list(self) -> "NDArray":
        r"""The dimensionless scaled wavefunction w(z) = z^{-1/2} \tilde{u}(x=z^2) = (r/a_0)^{-1/4} sqrt(a_0) r R(r)."""
        if self._w_list is None:
            return self.integrate()
        return self._w_list

    @property
    def u_list(self) -> "NDArray":
        r"""The dimensionless wavefunction \tilde{u}(x) = sqrt(a_0) r R(r)."""
        return np.sqrt(self.grid.z_list) * self.w_list

    @property
    def r_list(self) -> "NDArray":
        r"""The radial wavefunction R(r) in atomic units."""
        return self.u_list / self.grid.x_list

    def integrate(self, run_backward: bool = True, w0: float = 1e-10, _use_njit: bool = True) -> "NDArray":
        r"""Run the Numerov integration of the radial Schrödinger equation.

        The resulting radial wavefunctions are then stored as attributes, where
        - w_list is the dimensionless and scaled wavefunction w(z)
        - u_list is the dimensionless wavefunction \tilde{u}(x)
        - r_list is the radial wavefunction R(r) in atomic units

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
        if self._w_list is not None:
            raise ValueError("The wavefunction was already integrated, you should not integrate it again.")

        # Note: Inside this method we use y and x like it is used in the numerov function
        # and not like in the rest of this class, i.e. y = w(z) and x = z
        grid = self.grid

        n, l, j = self.model_potential.n, self.model_potential.l, self.model_potential.j
        glist = (
            8
            * self.element.reduced_mass_factor
            * grid.z_list
            * grid.z_list
            * (
                self.element.calc_energy(n, l, j, unit="a.u.")
                - self.model_potential.calc_total_effective_potential(grid.x_list)
            )
        )

        if run_backward:
            # Note: n - l - 1 is the number of nodes of the radial wavefunction
            # Thus, the sign of the wavefunction at the outer boundary is (-1)^{(n - l - 1) % 2}
            y0, y1 = 0, (-1) ** ((n - l - 1) % 2) * w0
            x_start, x_stop, dx = grid.z_max, grid.z_min, -grid.dz
            g_list_directed = glist[::-1]
            # We set x_min to the classical turning point
            # after x_min is reached in the integration, the integration stops, as soon as it crosses the x-axis again
            # or it reaches a local minimum (thus going away from the x-axis)
            x_min = self.model_potential.calc_z_turning_point("classical", dz=grid.dz)
            x_min = max(x_min, 5 * abs(dx), self.get_x_min())

        else:  # forward
            y0, y1 = 0, w0
            x_start, x_stop, dx = grid.z_min, grid.z_max, grid.dz
            g_list_directed = glist
            x_min = np.sqrt(self.model_potential.n * (self.model_potential.n + 15))

        if _use_njit:
            w_list_list = run_numerov_integration(x_start, x_stop, dx, y0, y1, g_list_directed, x_min)
        else:
            logger.warning("Using python implementation of Numerov integration, this is much slower!")
            w_list_list = _run_numerov_integration_python(x_start, x_stop, dx, y0, y1, g_list_directed, x_min)

        w_list = np.array(w_list_list)
        if run_backward:
            w_list = w_list[::-1]
            grid.set_grid_range(step_start=grid.steps - len(w_list))
        else:
            grid.set_grid_range(step_stop=len(w_list))

        # normalize the wavefunction, see docstring
        norm = np.sqrt(2 * np.sum(w_list * w_list * grid.z_list * grid.z_list) * grid.dz)
        w_list /= norm

        self._w_list = w_list

        self.sanity_check(x_stop, run_backward)
        return w_list

    def get_x_min(self) -> float:
        """Implement a few special cases for the x_min point of the integration."""
        species, n, l = self.element.species, self.model_potential.n, self.model_potential.l
        if species in ["Rb", "Cs"] and n == 4 and l == 3:
            return 2
        if species == "Sr_singlet" and n == 5 and l == 0:
            return 2

        return 0

    def sanity_check(self, z_stop: float, run_backward: bool) -> bool:  # noqa: C901, PLR0915, PLR0912
        """Do some sanity checks on the wavefunction.

        Check if the wavefuntion fulfills the following conditions:
        - The wavefunction is positive (or zero) at the inner boundary.
        - The wavefunction is close to zero at the inner boundary.
        - The wavefunction is close to zero at the outer boundary.
        - The wavefunction has exactly (n - l - 1) nodes.
        - The integration stopped before z_stop (for l>0)
        """
        warning_msgs = []

        grid = self.grid
        n = self.model_potential.n
        l = self.model_potential.l

        # Check and Correct if divergence of the wavefunction
        w_list_abs = np.abs(self.w_list)
        idmax = np.argmax(w_list_abs)
        w_abs_max = w_list_abs[idmax]
        outer_max = np.max(w_list_abs[int(0.1 * grid.steps) :])
        if idmax <= 5 and w_abs_max / outer_max > 10:
            warning_msgs.append(
                f"Wavefunction diverges at the inner boundary, w_abs_max / outer_max={w_abs_max / outer_max:.2e}",
            )
            warning_msgs.append("Trying to correct the wavefunction.")
            first_ind = np.argwhere(w_list_abs < outer_max)[0][0]
            self._w_list = self._w_list[first_ind:]  # type: ignore [index]
            grid.set_grid_range(step_start=first_ind)
            norm = np.sqrt(2 * np.sum(self.w_list * self.w_list * grid.z_list * grid.z_list) * grid.dz)
            self._w_list /= norm

        # Check the maximum of the wavefunction
        idmax = np.argmax(np.abs(self.w_list))
        if idmax < 0.05 * grid.steps:
            warning_msgs.append(
                f"The maximum of the wavefunction is close to the inner boundary (idmax={idmax}) "
                "probably due to inner divergence of the wavefunction. "
            )

        # Check the wavefunction at the inner boundary
        if self.w_list[0] < 0:
            warning_msgs.append(f"The wavefunction is negative at the inner boundary ({self.w_list[0]}).")

        # Check the weight of the wavefunction at the inner boundary
        inner_ind = 10
        inner_weight = (
            2
            * np.sum(
                self.w_list[:inner_ind] * self.w_list[:inner_ind] * grid.z_list[:inner_ind] * grid.z_list[:inner_ind]
            )
            * grid.dz
        )
        inner_weight_scaled_to_whole_grid = inner_weight * grid.steps / inner_ind

        tol = 1e-4
        if n <= 11:
            # for low n the wavefunction converges not as good and still has more weight at the inner boundary
            tol = 5e-3

        if inner_weight_scaled_to_whole_grid > tol:
            warning_msgs.append(
                f"The wavefunction is not close to zero at the inner boundary"
                f" (inner_weight_scaled_to_whole_grid={inner_weight_scaled_to_whole_grid:.2e})"
            )

        # Check the wavefunction at the outer boundary
        outer_ind = int(0.95 * grid.steps)
        outer_wf = self.w_list[outer_ind:]
        if np.mean(outer_wf) > 1e-7:
            warning_msgs.append(
                f"The wavefunction is not close to zero at the outer boundary, mean={np.mean(outer_wf):.2e}"
            )

        outer_weight = 2 * np.sum(outer_wf * outer_wf * grid.z_list[outer_ind:] * grid.z_list[outer_ind:]) * grid.dz
        outer_weight_scaled_to_whole_grid = outer_weight * grid.steps / len(outer_wf)
        if outer_weight_scaled_to_whole_grid > 1e-10:
            warning_msgs.append(
                f"The wavefunction is not close to zero at the outer boundary,"
                f" (outer_weight_scaled_to_whole_grid={outer_weight_scaled_to_whole_grid:.2e})"
            )

        # Check the number of nodes
        nodes = np.sum(np.abs(np.diff(np.sign(self.w_list)))) // 2
        if nodes != n - l - 1:
            warning_msgs.append(f"The wavefunction has {nodes} nodes, but should have {n - l - 1} nodes.")

        # Check that numerov stopped and did not run until x_stop
        if l > 0:
            if run_backward and z_stop > grid.z_list[0] - grid.dz / 2:
                warning_msgs.append(f"The integration did not stop before z_stop, z={grid.z_list[0]}, z_stop={z_stop}")
            if not run_backward and z_stop < grid.z_list[-1] + grid.dz / 2:
                warning_msgs.append(f"The integration did not stop before z_stop, z={grid.z_list[-1]}")
        elif l == 0 and run_backward:
            if z_stop > 1.5 * grid.dz:
                warning_msgs.append(f"The integration for l=0 should go until z=dz, but a z_stop={z_stop} was used.")
            elif grid.z_list[0] > 2.5 * grid.dz:
                # z_list[0] should be dz, but if it is 2 * dz this is also fine
                # e.g. this might happen if the integration just stopped at the last step due to a negative y value
                warning_msgs.append(
                    f"The integration for l=0 did stop before the z_min boundary, z={grid.z_list[0]}, {grid.dz}"
                )

        if warning_msgs:
            species, j = self.element.species, self.model_potential.j
            msg = f"The wavefunction (species={species} n={n}, l={l}, j={j:.1f}) has some issues:"
            msg += "\n      ".join(["", *warning_msgs])
            logger.warning(msg)
            return False

        return True
