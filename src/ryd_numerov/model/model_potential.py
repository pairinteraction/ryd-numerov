import logging
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

from ryd_numerov.units import ureg

if TYPE_CHECKING:
    from ryd_numerov.elements import BaseElement
    from ryd_numerov.units import NDArray


logger = logging.getLogger(__name__)

ADDITIONAL_POTENTIALS = Literal["spin_orbit", "core_corrections", "core_polarization"]


class ModelPotential:
    """Model potential parameters for an atomic species and angular momentum.

    Attributes:
        species: Atomic species
        n: Principal quantum number
        l: Orbital angular momentum quantum number
        s: Spin quantum number
        j: Total angular momentum quantum number
        ac: Polarizability parameter in atomic units.
        Z: Nuclear charge.
        a1: Model potential parameter a1 in atomic units.
        a2: Model potential parameter a2 in atomic units.
        a3: Model potential parameter a3 in atomic units.
        a4: Model potential parameter a4 in atomic units.
        rc: Core radius parameter in atomic units.


    """

    def __init__(
        self,
        element: "BaseElement",
        n: int,
        l: int,
        s: float,
        j: float,
        additional_potentials: Optional[list[ADDITIONAL_POTENTIALS]] = None,
    ) -> None:
        r"""Initialize the model potential.

        Args:
            element: BaseElement object representing the atomic species.
            n: Principal quantum number
            l: Orbital angular momentum quantum number
            s: Spin quantum number
            j: Total angular momentum quantum number
            additional_potentials: List of additional potentials to include in the model.

        """
        self.element = element
        self.n = n
        self.l = l
        self.s = s
        self.j = j

        if additional_potentials is None:
            additional_potentials = self.element.additional_potentials_default
        self.additional_potentials = additional_potentials

    def calc_potential_coulomb(self, x: "NDArray") -> "NDArray":
        r"""Calculate the coulomb potential V_Col(x) in atomic units.

        The coulomb potential is given as

        .. math::
            V_{Col}(x) = -1 / x

        where x = r / a_0.

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_Col: The coulomb potential V_Col(x) in atomic units.

        """
        return -1 / x

    def calc_potential_core_corrections(self, x: "NDArray") -> "NDArray":
        r"""Calculate the core potential corrections V_cc(x) in atomic units.

        The core potential is given as

        .. math::
            V_cc(x) = -(Z_{nl}-1) / x

        where x = r / a_0 and Z_{nl} is the effective nuclear charge

        .. math::
            Z_{nl} = 1 + (Z - 1) \exp(-a_1 x) - x (a_3 + a_4 x) \exp(-a_2 x)

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_cc: The core potential corrections V_cc(x) in atomic units.

        """
        a1, a2, a3, a4 = self.element.get_parametric_model_potential_parameters(self.l)
        exp_a1 = np.exp(-a1 * x)
        exp_a2 = np.exp(-a2 * x)
        z_nl: NDArray = (self.element.Z - 1) * exp_a1 - x * (a3 + a4 * x) * exp_a2
        return -z_nl / x

    def calc_potential_core_polarization(self, x: "NDArray") -> "NDArray":
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
        alpha_c = self.element.alpha_c
        if alpha_c == 0:
            return np.zeros_like(x)
        x_c = self.element.get_r_c(self.l)
        x2: NDArray = x * x
        x4: NDArray = x2 * x2
        x6: NDArray = x4 * x2
        exp_x6 = np.exp(-(x6 / x_c**6))
        v_p: NDArray = -alpha_c / (2 * x4) * (1 - exp_x6)
        return v_p

    def calc_potential_spin_orbit(self, x: "NDArray") -> "NDArray":
        r"""Calculate the spin-orbit coupling potential V_so(x) in atomic units.

        The spin-orbit coupling potential is given as

        .. math::
            V_{so}(x > x_c) = \frac{\alpha^2}{4x^3} [j(j+1) - l(l+1) - s(s+1)]

        where x = r / a_0, \alpha is the fine structure constant,
        j is the total angular momentum quantum number, l is the orbital angular momentum
        quantum number, and s is the spin quantum number.

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_so: The spin-orbit coupling potential V_so(x) in atomic units.

        """
        alpha = ureg.Quantity(1, "fine_structure_constant").to_base_units().magnitude
        x3 = x * x * x
        v_so: NDArray = alpha**2 / (4 * x3) * (self.j * (self.j + 1) - self.l * (self.l + 1) - self.s * (self.s + 1))
        x_c = self.element.get_r_c(self.l)
        if x[0] < x_c:
            v_so *= x > x_c
        return v_so

    def calc_potential_centrifugal(self, x: "NDArray") -> "NDArray":
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
        x2 = x * x
        return (1 / self.element.reduced_mass_factor) * self.l * (self.l + 1) / (2 * x2)

    def calc_effective_potential_sqrt(self, x: "NDArray") -> "NDArray":
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
        x2 = x * x
        return (1 / self.element.reduced_mass_factor) * (3 / 32) / x2

    def calc_total_physical_potential(self, x: "NDArray") -> "NDArray":
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
        v_tot = self.calc_potential_coulomb(x) + self.calc_potential_centrifugal(x)
        if "spin_orbit" in self.additional_potentials:
            v_tot += self.calc_potential_spin_orbit(x)
        if "core_corrections" in self.additional_potentials:
            v_tot += self.calc_potential_core_corrections(x)
        if "core_polarization" in self.additional_potentials:
            v_tot += self.calc_potential_core_polarization(x)

        return v_tot

    def calc_total_effective_potential(self, x: "NDArray") -> "NDArray":
        r"""Calculate the total potential V_tot(x) in atomic units.

        The total effective potential includes all physical and non-physical potentials:

        .. math::
            V_{tot}(x) = V_c(x) + V_p(x) + V_l(x) + V_{so}(x) + V_{sqrt}(x)

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_tot: The total potential V_tot(x) in atomic units.

        """
        return self.calc_total_physical_potential(x) + self.calc_effective_potential_sqrt(x)

    def calc_z_turning_point(self, which: Literal["hydrogen", "classical", "zerocrossing"], dz: float = 1e-3) -> float:
        r"""Calculate the inner turning point z_i for the model potential.

        There are three different turning points we consider:

        - The hydrogen turning point, where for the idealized hydrogen atom the potential equals the energy,
          i.e. V_c(r_i) + V_l(r_i) = E.
          This is exactly the case at

            .. math::
                r_i = n^2 - n \sqrt{n^2 - l(l + 1)}

        - The classical turning point, where the physical potential of the Rydberg model potential equals the energy,
          i.e. V_phys(r_i) = V_c(r_i) + V_p(r_i) + V_l(r_i) + V_{so}(r_i) = E.

        - The zero-crossing turning point, where the physical potential of the Rydberg model potential equals zero,
          i.e. V_phys(r_i) = V_c(r_i) + V_p(r_i) + V_l(r_i) + V_{so}(r_i) = 0.

        Args:
            which: Which turning point to calculate, one of "hydrogen", "classical", "zerocrossing".
            dz: The precision of the turning point calculation.

        Returns:
            z_i: The inner turning point z_i in the scaled dimensionless coordinate z_i = sqrt{r_i / a_0}.

        """
        assert which in ["hydrogen", "classical", "zerocrossing"], f"Invalid turning point method {which}."
        hydrogen_r_i: float = self.n * self.n - self.n * np.sqrt(self.n * self.n - self.l * (self.l - 1))
        hydrogen_z_i: float = np.sqrt(hydrogen_r_i)

        if which == "hydrogen":
            return hydrogen_z_i

        if which == "classical":
            z_list = np.arange(max(dz, hydrogen_z_i - 10), hydrogen_z_i + 10, dz)
            energy = self.element.calc_energy(self.n, self.l, self.j, unit="a.u.")
        elif which == "zerocrossing":
            z_list = np.arange(max(dz, hydrogen_z_i / 2 - 5), hydrogen_z_i + 10, dz)
            energy = 0

        x_list = z_list * z_list
        v_phys = self.calc_total_physical_potential(x_list)
        arg: int = np.argwhere(v_phys < energy)[0][0]

        if arg == 0:
            if self.l == 0:
                return 0
            logger.warning("Turning point is at arg=0, this shouldnt happen.")
        elif arg == len(z_list) - 1:
            logger.warning("Turning point is at maixmal arg, this shouldnt happen.")

        return z_list[arg]  # type: ignore [no-any-return]  # FIXME: numpy indexing
