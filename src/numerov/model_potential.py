from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Optional

import numpy as np

from numerov.database import QuantumDefectsDatabase
from numerov.units import ureg

if TYPE_CHECKING:
    from numerov.rydberg import RydbergState


@dataclass
class ModelPotential:
    """A class to represent the Rydberg model potential.

    All parameters and potentials are in atomic units.
    """

    state: "RydbergState"
    qdd_path: Optional[str] = None
    add_spin_orbit: bool = True

    def __post_init__(self) -> None:
        self.load_parameters_from_database()

    def load_parameters_from_database(self) -> None:
        """Load the model potential and Rydberg-Ritz parameters from the QuantumDefectsDatabase.

        For more details see `database.QuantumDefectsDatabase`.
        """
        state = self.state
        self.qdd = QuantumDefectsDatabase(self.qdd_path)

        self.model_params = self.qdd.get_model_potential(state.species, state.l)
        self.ritz_params = self.qdd.get_rydberg_ritz(state.species, state.l, state.j)

    def calc_V_c(self, x: np.ndarray) -> np.ndarray:
        r"""Calculate the core potential V_c(x) in atomic units.

        The core potential is given as

        .. math::
            V_c(x) = -Z_{nl} / x

        where x = r / a_0 and Z_{nl} is the effective nuclear charge

        .. math::
            Z_{nl} = 1 + (Z - 1) \exp(-a_1 x) - x (a_3 + a_4 x) \exp(-a_2 x)

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_c: The core potential V_c(x) in atomic units.

        """
        # if self.state.l >= 4:  # TODO check if we want this
        #     return -1 / x
        params = self.model_params
        Z_nl = 1 + (params.Z - 1) * np.exp(-params.a1 * x) - x * (params.a3 + params.a4 * x) * np.exp(-params.a2 * x)
        V_c = -Z_nl / x
        return V_c

    def calc_V_p(self, x: np.ndarray) -> np.ndarray:
        r"""Calculate the core polarization potential V_p(x) in atomic units.

        The core polarization potential is given as

        .. math::
            V_p(x) = -\frac{a_c}{2x^4} (1 - e^{-x^6/x_c**6})

        where x = r / a_0, a_c is the static core dipole polarizability and x_c is the effective core size.

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_p: The polarization potential V_p(x) in atomic units.

        """
        params = self.model_params
        if params.ac == 0:  # TODO or l >= 4 like arc?
            return np.zeros_like(x)
        V_p = -params.ac / (2 * x**4) * (1 - np.exp(-((x / params.xc) ** 6)))
        return V_p

    def calc_V_so(self, x: np.ndarray) -> np.ndarray:
        r"""Calculate the spin-orbit coupling potential V_so(x) in atomic units.

        The spin-orbit coupling potential is given as

        .. math::
            V_{so}(x) = \frac{\alpha}{4x^3} [j(j+1) - l(l+1) - s(s+1)]

        where x = r / a_0, \alpha is the fine structure constant,
        j is the total angular momentum quantum number, l is the orbital angular momentum
        quantum number, and s is the spin quantum number.

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_so: The spin-orbit coupling potential V_so(x) in atomic units.

        """
        # TODO pairinteraction old:
        # # if self.l > 4 then return 0?
        # # alpha * alpha isntead of one alpha
        # TODO ARC
        # # if x[0] < self.xc return 0 (for those x)
        alpha = ureg.Quantity(1, "fine_structure_constant").to_base_units().magnitude
        V_so = (
            alpha**2
            / (4 * x**3)
            * (
                self.state.j * (self.state.j + 1)
                - self.state.l * (self.state.l + 1)
                - self.state.s * (self.state.s + 1)
            )
        )  # TODO alpha**2 from arc
        return V_so

    def calc_V_l(self, x: np.ndarray) -> np.ndarray:
        r"""Calculate the centrifugal potential V_l(x) in atomic units.

        The centrifugal potential is given as

        .. math::
            V_l(x) = \frac{l(l+1)}{2x^2}

        where x = r / a_0 and l is the orbital angular momentum quantum number.

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_l: The centrifugal potential V_l(x) in atomic units.

        """
        V_l = self.state.l * (self.state.l + 1) / (2 * x**2)
        return V_l

    def calc_V_sqrt(self, x: np.ndarray) -> np.ndarray:
        r"""Calculate the effective potential V_sqrt(x) from the sqrt transformation in atomic units.

        The sqrt transformation potential arises from the transformation from the wavefunction u(x) to w(z),
        where x = r / a_0 and w(z) = z^{-1/2} u(x=z^2) = (r/a_0)^{-1/4} sqrt(a_0) r R(r).
        Due to the transformation, an additional term is added to the radial Schrödinger equation,
        which can be written as effective potential V_{sqrt}(x) and is given by

        .. math::
            V_{sqrt}(x) = \frac{3}{32x^2}

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_sqrt: The sqrt transformation potential V_sqrt(x) in atomic units.

        """
        V_sqrt = (3 / 32) / x**2
        return V_sqrt

    def calc_V_phys(self, x: np.ndarray) -> np.ndarray:
        r"""Calculate the total physical potential V_phys(x) in atomic units.

        The total physical potential is the sum of the core potential, polarization potential,
        centrifugal potential, and optionally the spin-orbit coupling:

        .. math::
            V_{phys}(x) = V_c(x) + V_p(x) + V_l(x) + V_{so}(x)

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_phys: The total physical potential V_phys(x) in atomic units.

        """
        V_tot = self.calc_V_c(x) + self.calc_V_p(x) + self.calc_V_l(x)
        if self.add_spin_orbit:
            V_tot += self.calc_V_so(x)
        return V_tot

    def calc_V_tot(self, x: np.ndarray) -> np.ndarray:
        r"""Calculate the total potential V_tot(x) in atomic units.

        The total effective potential includes all physical and non-physical potentials:

        .. math::
            V_{tot}(x) = V_c(x) + V_p(x) + V_l(x) + V_{so}(x) + V_{sqrt}(x)

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_tot: The total potential V_tot(x) in atomic units.

        """
        V_tot = self.calc_V_phys(x) + self.calc_V_sqrt(x)
        return V_tot

    @cached_property
    def energy(self) -> float:
        r"""Return the energy of a Rydberg state with principal quantum number n in atomic units.

        The effective principal quantum number in quantum defect theory is defined as series expansion

        .. math::
            n^* = n - \\delta_{nlj}

        where

        .. math::
            \\delta_{nlj} = d_0 + \frac{d_2}{(n - d_0)^2} + \frac{d_4}{(n - d_0)^4} + \frac{d_6}{(n - d_0)^6}

        is the quantum defect. The energy of the Rydberg state is then given by

        .. math::
            E_{nlj} / E_H = -\frac{1}{2} \frac{Ry}{Ry_\\infty} \frac{1}{n^*}

        where :math:`E_H` is the Hartree energy (the atomic unit of energy).

        Args:
            n: Principal quantum number of the state to calculate the energy for.

        Returns:
            Energy of the Rydberg state in atomic units.

        """
        params = self.ritz_params
        state = self.state
        Ry_inf = ureg.Quantity(1, "rydberg_constant").to("1/cm").magnitude
        delta_nlj = (
            params.d0
            + params.d2 / (state.n - params.d0) ** 2
            + params.d4 / (state.n - params.d0) ** 4
            + params.d6 / (state.n - params.d0) ** 6
        )
        nstar = state.n - delta_nlj
        E_nlj = -0.5 * (params.Ry / Ry_inf) / nstar**2
        return E_nlj
