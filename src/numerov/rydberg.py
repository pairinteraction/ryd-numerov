import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, TypeVar, Union

import numpy as np

from numerov.integration import _python_run_numerov_integration, run_numerov_integration

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
        in dimensionless units.
        xmax (default see `RydbergState.set_range`): The maximum value of the radial coordinate
        in dimensionless units.
        dx (default see `RydbergState.set_range`): The step size of the integration in dimensionless units
        (corresponds to h in the equation above).
        steps (default see `RydbergState.set_range`): The number of steps of the integration (use either steps or dx).
        parameter_dict (default: None): A dictionary containing the parameters for the effective potential.
        If not provided, the parameters are loaded from the database.
    """

    species: str
    n: int
    l: int
    j: float

    run_backward: bool = True
    epsilon_u: float = 1e-10
    _use_njit: bool = True

    xmin: Optional[float] = None
    xmax: Optional[float] = None
    dx: Optional[float] = None
    steps: Optional[int] = None

    parameter_dict: Optional[dict[str, float]] = None

    def __post_init__(self) -> None:
        self.s: Union[int, float]
        if self.species.endswith("singlet"):
            self.s = 0
        elif self.species.endswith("triplet"):
            self.s = 1
        else:
            self.s = 0.5

        assert self.l <= self.n - 1, "l must be smaller than n - 1"
        assert self.j >= abs(self.l - self.s) and self.j <= self.l + self.s, "j must be between l - s and l + s"

        if self.parameter_dict is not None:
            self._load_parameters_from_dict()
        elif self.species in ["H", "He+"]:
            self._load_hydrogen_like_parameters()
        else:
            self._load_parameters_from_database()
        self.set_range()

    def _load_parameters_from_dict(self) -> None:
        assert self.parameter_dict is not None
        self.Z = self.parameter_dict["Z"]
        self.energy = self.parameter_dict["energy"]
        self.a1 = self.parameter_dict["a1"]
        self.a2 = self.parameter_dict["a2"]
        self.a3 = self.parameter_dict["a3"]
        self.a4 = self.parameter_dict["a4"]
        self.alphac = self.parameter_dict["alphac"]
        self.xc = self.parameter_dict["xc"]

    def _load_parameters_from_database(self) -> None:
        self.Z = np.inf
        self.energy = np.inf
        self.a1 = self.a2 = self.a3 = self.a4 = np.inf
        self.alphac = np.inf
        self.xc = np.inf

    def _load_hydrogen_like_parameters(self) -> None:
        self.Z = {"H": 1, "He+": 2}[self.species]
        self.energy = -(self.Z**2) / (self.n**2)
        self.a1 = self.a2 = self.a3 = self.a4 = 0
        self.ac = 0
        self.xc = np.inf

        self.alpha1, self.alpha2, self.alpha3, self.alpha4 = self.a1, self.a2, self.a3, self.a4
        self.zc = self.xc
        self.alphac = self.ac

    def set_range(self) -> None:
        """Automatically determine sensful default values for xmin, xmax and dx.

        The x-values represent the raidal coordinate in units of the Bohr radius a_0 (x = r / a_0).
        Furthermore, we define z-values as z = sqrt(x) = sqrt(r / a_0), which is used for the integration.
        The benefit of using z is that the nodes of the wavefunction are equally spaced in z-space,
        allowing for a computational better choice of choosing the constant step size during the integration.


        """
        if self.xmin is None:
            if not self.run_backward or self.l == 0:
                self.xmin = 1e-5
            else:  # self.run_backward
                xmin = self.n * self.n - self.n * np.sqrt(self.n * self.n - (self.l - 1) * (self.l - 1))
                self.xmin = max(0.1, xmin)
        if self.xmax is None:
            if self.run_backward:
                self.xmax = 2 * self.n * (self.n + 25)
            else:
                self.xmax = 2 * self.n * (self.n + 6)
        zmin, zmax = np.sqrt(self.xmin), np.sqrt(self.xmax)

        if self.dx is not None and self.steps is not None:
            raise ValueError("Use either dx or steps, not both.")
        elif self.dx is None:
            if self.steps is None:
                self.steps = 10_000
            self.z_list = np.linspace(zmin, zmax, self.steps, endpoint=True)
        elif self.dx is not None:
            dz = np.sqrt(self.dx)
            self.z_list = np.arange(zmin, zmax + dz, dz)

        self.dz = self.z_list[1] - self.z_list[0]
        self.steps = len(self.z_list)

        self.x_list = np.power(self.z_list, 2)

    @cached_property
    def V_tot(self) -> np.ndarray:
        z = self.z_list
        z2 = z**2
        z4 = z**4
        z6 = z**6
        z8 = z**8

        Z_nl = 1 + (self.Z - 1) * np.exp(-self.a1 * z2) - z2 * (self.a3 + self.a4 * z2) * np.exp(-self.a2 * z2)
        V_c = -Z_nl / z2 * 2  # TODO check 2 in diemnsionless units
        V_p = -self.alphac / (2 * z8) * (1 - np.exp(-((z / self.zc) ** 12)))
        if self.species in ["H", "He+"]:  # TODO or self.l > 4 ???
            V_so = 0
        else:
            alpha = 1 / 137.035999084
            V_so = (
                alpha / (4 * z6) * (self.j * (self.j + 1) - self.l * (self.l + 1) - self.s * (self.s + 1))
            )  # TODO check wheter 1 or 1/4?
        V_l = self.l * (self.l + 1) / z4
        V_sqrt = (3 / 16) / z4
        return V_c + V_p + V_so + V_l + V_sqrt

    @cached_property
    def g_list(self) -> np.ndarray:
        return 4 * self.z_list**2 * (self.energy - self.V_tot)

    def integrate(self) -> None:
        r"""Run the Numerov integration of the radial Schrödinger equation for the desired state,

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
            # raise ValueError(f"xmax={self.xmax} was chosen too small ({sum_large_z=}), increase xmax.")

        self.u_list = np.sqrt(self.z_list) * self.w_list
        self.R_list = self.u_list / self.x_list
