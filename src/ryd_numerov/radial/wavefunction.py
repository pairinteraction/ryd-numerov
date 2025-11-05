from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from mpmath import whitw
from scipy.special import gamma

from ryd_numerov.radial.numerov import _run_numerov_integration_python, run_numerov_integration
from ryd_numerov.species.species_object import BaseElement

if TYPE_CHECKING:
    from ryd_numerov.radial import Grid, Model
    from ryd_numerov.radial.radial_state import RadialState
    from ryd_numerov.units import NDArray

logger = logging.getLogger(__name__)

WavefunctionSignConvention = Optional[Literal["positive_at_outer_bound", "n_l_1"]]


class Wavefunction(ABC):
    r"""An object containing all the relevant information about the radial wavefunction."""

    def __init__(
        self,
        radial_state: RadialState,
        grid: Grid,
    ) -> None:
        """Create a Wavefunction object.

        Args:
            radial_state: The RadialState object.
            grid: The grid object.

        """
        self.radial_state = radial_state
        self.grid = grid

        self._w_list: NDArray | None = None

    @property
    def w_list(self) -> NDArray:
        r"""The dimensionless scaled wavefunction w(z) = z^{-1/2} \tilde{u}(x=z^2) = (r/a_0)^{-1/4} sqrt(a_0) r R(r)."""
        if self._w_list is None:
            self.integrate()
            assert self._w_list is not None
        return self._w_list

    @property
    def u_list(self) -> NDArray:
        r"""The dimensionless wavefunction \tilde{u}(x) = sqrt(a_0) r R(r)."""
        return np.sqrt(self.grid.z_list) * self.w_list

    @property
    def r_list(self) -> NDArray:
        r"""The radial wavefunction \tilde{R}(r) in atomic units \tilde{R}(r) = a_0^{-3/2} R(r)."""
        return self.u_list / self.grid.x_list

    @property
    def nodes(self) -> int:
        """The number of nodes (i.e. zero-crossings) of the wavefunction."""
        return int(np.sum(np.abs(np.diff(np.sign(self.w_list)))) // 2)

    @abstractmethod
    def integrate(self) -> None:
        """Integrate the radial Schrödinger equation and store the wavefunction in the w_list attribute."""

    def apply_sign_convention(self, sign_convention: WavefunctionSignConvention) -> None:
        """Set the sign of the wavefunction according to the sign convention.

        Args:
            sign_convention: The sign convention for the wavefunction.
                - None: Leave the wavefunction as it is.
                - "n_l_1": The wavefunction is defined to have the sign of (-1)^{(n - l - 1)} at the outer boundary.
                - "positive_at_outer_bound": The wavefunction is defined to be positive at the outer boundary.

        """
        if self._w_list is None:
            raise ValueError("The wavefunction has not been integrated yet.")

        if sign_convention is None:
            return

        current_outer_sign = 1
        for w in self._w_list[::-1]:
            if w != 0 and not np.isnan(w):
                current_outer_sign = np.sign(w)
                break

        if sign_convention == "n_l_1":
            assert self.radial_state.n is not None, "n must be given to apply the n_l_1 sign convention."
            if current_outer_sign != (-1) ** (self.radial_state.n - self.radial_state.l_r - 1):
                self._w_list = -self._w_list
        elif sign_convention == "positive_at_outer_bound":
            if current_outer_sign != 1:
                self._w_list = -self._w_list
        else:
            raise ValueError(f"Unknown sign convention: {sign_convention}")


class WavefunctionNumerov(Wavefunction):
    r"""A wavefunction object for the Numerov method.

    We solve the radial dimensionless Schrödinger equation for the Rydberg state

    .. math::
        \frac{d^2}{dx^2} u(x) = - \\left[ E - V_{eff}(x) \right] u(x)

    using the Numerov method, see `integration.run_numerov_integration`.
    """

    def __init__(
        self,
        radial_state: RadialState,
        grid: Grid,
        model: Model,
    ) -> None:
        """Create a Wavefunction object.

        Args:
            radial_state: The RadialState object.
            grid: The grid object.
            model: The model object.

        """
        super().__init__(radial_state, grid)
        self.model = model

    def integrate(self, run_backward: bool = True, w0: float = 1e-10, *, _use_njit: bool = True) -> None:
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

        element = BaseElement.from_species(self.radial_state.species)
        energy_au = element.calc_energy_from_nu(self.radial_state.nu)
        v_eff = self.model.calc_total_effective_potential(grid.x_list)
        glist = 8 * element.reduced_mass_factor * grid.z_list * grid.z_list * (energy_au - v_eff)

        if run_backward:
            # During the Numerov integration we define the wavefunction such that it should always stop
            # at the inner boundary with positive weight
            # Note: n - l - 1 is the number of nodes of the radial wavefunction
            # Thus, the sign of the wavefunction at the outer boundary is (-1)^{(n - l - 1) % 2}
            # You can choose a different sign convention by calling the method apply_sign_convention() afterwards.
            y0, y1 = 0, w0
            x_start, x_stop, dx = grid.z_max, grid.z_min, -grid.dz
            g_list_directed = glist[::-1]
            # We set x_min to the classical turning point
            # after x_min is reached in the integration, the integration stops, as soon as it crosses the x-axis again
            # or it reaches a local minimum (thus going away from the x-axis)
            # the reason for this is that the second derivative of the wavefunction d^2/dz^2 w(z) (= concavity)
            # can only vanish at either
            # i) where w(z) = 0 or ii) where the potential is equal to the energy (-> classical turning point)
            # If we further assume, that the wavefunction converges to zero at the inner boundary,
            # we know that after the inner classical turning point
            # the wavefunction should never increase the distance from the x-axis again.
            x_min = self.model.calc_turning_point_z(energy_au)

        else:  # forward
            y0, y1 = 0, w0
            x_start, x_stop, dx = grid.z_min, grid.z_max, grid.dz
            g_list_directed = glist
            n = self.radial_state.n if self.radial_state.n is not None else self.radial_state.nu
            x_min = math.sqrt(n * (n + 15))

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
        norm = math.sqrt(2 * np.sum(w_list * w_list * grid.z_list * grid.z_list) * grid.dz)
        w_list /= norm

        self._w_list = w_list

        self.sanity_check(x_stop, run_backward)

    def sanity_check(self, z_stop: float, run_backward: bool) -> bool:  # noqa: C901, PLR0915, PLR0912
        """Do some sanity checks on the wavefunction.

        Check if the wavefuntion fulfills the following conditions:
        - The wavefunction is positive (or zero) at the inner boundary.
        - The wavefunction is close to zero at the inner boundary.
        - The wavefunction is close to zero at the outer boundary.
        - The wavefunction has exactly (n - l - 1) nodes.
        - The integration stopped before z_stop (for l>0)
        """
        warning_msgs: list[str] = []

        grid = self.grid
        state = self.radial_state

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
            norm = math.sqrt(2 * np.sum(self.w_list * self.w_list * grid.z_list * grid.z_list) * grid.dz)
            self._w_list /= norm

        # Check the maximum of the wavefunction
        idmax = np.argmax(np.abs(self.w_list))
        if idmax < 0.05 * grid.steps:
            warning_msgs.append(
                f"The maximum of the wavefunction is close to the inner boundary (idmax={idmax}) "
                "probably due to inner divergence of the wavefunction. "
            )

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
        # for low n the wavefunction converges not as good and still has more weight at the inner boundary
        n = state.n if state.n is not None else state.nu + 5
        if n <= 10:
            tol = 8e-3
        elif n <= 16:
            tol = 2e-3

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
        nodes = self.nodes
        if state.n is not None and nodes != state.n - state.l_r - 1:
            warning_msgs.append(f"The wavefunction has {nodes} nodes, but should have {state.n - state.l_r - 1} nodes.")

        # Check that numerov stopped and did not run until x_stop
        if state.l_r > 0:
            if run_backward and z_stop > grid.z_list[0] - grid.dz / 2 and inner_weight_scaled_to_whole_grid > 1e-6:
                warning_msgs.append(f"The integration did not stop before z_stop, z={grid.z_list[0]}, z_stop={z_stop}")
            if not run_backward and z_stop < grid.z_list[-1] + grid.dz / 2:
                warning_msgs.append(f"The integration did not stop before z_stop, z={grid.z_list[-1]}")
        elif state.l_r == 0 and run_backward:
            if grid.z_list[0] > 0.035:  # z_list[0] should run almost to zero for l=0
                warning_msgs.append(f"The integration for l=0 did stop at {grid.z_list[0]} (should be close to zero).")

        if warning_msgs:
            msg = f"The wavefunction for the state {state} has some issues:"
            msg += "\n      ".join(["", *warning_msgs])
            logger.warning(msg)
            return False

        return True


class WavefunctionWhittaker(Wavefunction):
    def integrate(self) -> None:
        logger.warning("Using Whittaker to get the wavefunction is not recommended! Use this only for comparison.")
        l = self.radial_state.l_r
        nu = self.radial_state.nu

        whitw_vectorized = np.vectorize(whitw, otypes=[float])
        whitw_list = whitw_vectorized(nu, l + 0.5, 2 * self.grid.x_list / nu)

        u_list: NDArray = whitw_list / np.sqrt(nu**2 * gamma(nu + l + 1) * gamma(nu - l))
        w_list: NDArray = u_list / np.sqrt(self.grid.z_list)

        self._w_list = w_list
