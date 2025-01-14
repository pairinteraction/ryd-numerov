from dataclasses import dataclass
from typing import TypeVar, Union

import numpy as np

from numerov.integration import _python_run_numerov_integration, run_numerov_integration

ValueType = TypeVar("ValueType", bound=Union[float, np.ndarray])


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
            for backward integration we set u[-1] = 0 and u[-2] = (-1)^{(n - l + 1) % 2} * epsilon_u.
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
        if self.species.endswith("singlet"):
            self.s = 0
        elif self.species.endswith("triplet"):
            self.s = 1
        else:
            self.s = 0.5

        assert self.l <= self.n - 1, "l must be smaller than n - 1"
        assert self.j >= abs(self.l - self.s) and self.j <= self.l + self.s, "j must be between l - s and l + s"

        if self.species in ["H", "He+"]:
            self._load_hydrogen_like_parameters()
        else:
            self._load_parameters_from_database()
        self._auto_set_range()

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
        self.alphac = 0
        self.xc = np.inf

    def _auto_set_range(self) -> None:
        """dx: The step size of the integration in dimensionless units (corresponds to h in the equation above).
        xmin: The minimum value of the radial coordinate in dimensionless units.
        xmax: The maximum value of the radial coordinate in dimensionless units."""
        self.dx = 1e-3
        self.xmin = self.dx
        self.xmax = 80

    def calc_Veff(self, x: ValueType) -> ValueType:
        # TODO check units
        # TODO add pint quantities, e.g. for alpha, ...
        Z_nl = 1 + (self.Z - 1) * np.exp(-self.a1 * x) - x * (self.a3 + self.a4 * x) * np.exp(-self.a2 * x)
        V_c = -2 * Z_nl / x  # TODO check 2 in diemnsionless units
        V_p = -self.alphac / (2 * x**4) * (1 - np.exp(-((x / self.xc) ** 6)))
        if self.species in ["H", "He+"]:
            V_SO = 0
        else:
            alpha = 1 / 137.035999084
            V_SO = (
                alpha / (4 * x**3) * (self.j * (self.j + 1) - self.l * (self.l + 1) - self.s * (self.s + 1))
            )  # TODO check wheter 1 or 1/4?
        V_l = self.l * (self.l + 1) / (x**2)
        Veff = V_c + V_p + V_SO + V_l
        return Veff

    def calc_g_list(self, x: ValueType) -> ValueType:
        return self.energy - self.calc_Veff(x)

    def integrate(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Run the Numerov integration of the radial Schrödinger equation for the desired state,

        Returns:
            x_list: A numpy array of the values of the radial coordinate at which the wavefunction was evaluated.
            u_list: A numpy array of the values of the radial wavefunction at each value of x_list.
                The radial wavefunction is normalized such that

                .. math::
                    \int_{0}^{\infty} x^2 |R(x)|^2 dx = \int_{0}^{\infty} |u(x)|^2 dx = 1
        """
        self.x_list = x_list = np.arange(self.xmin, self.xmax + self.dx, self.dx)
        self.g_list = g_list = self.calc_g_list(x_list)

        if self.run_backward:
            y0, y1 = 0, (-1) ** ((self.n - self.l + 1) % 2) * self.epsilon_u
        else:  # forward
            y0, y1 = 0, self.epsilon_u

        if self._use_njit:
            self.u_list = run_numerov_integration(self.dx, len(x_list), y0, y1, g_list, self.run_backward)
        else:
            self.u_list = _python_run_numerov_integration(self.dx, len(x_list), y0, y1, g_list, self.run_backward)

        # normalize the wavefunction, such that
        # \int_{0}^{\infty} x^2 |R(x)|^2 dx = \int_{0}^{\infty} |u(x)|^2 dx = 1
        norm = np.sqrt(np.sum(self.u_list**2) * self.dx)
        self.u_list /= norm

        return self.x_list, self.u_list
