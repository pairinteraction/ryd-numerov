import logging
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

from ryd_numerov.model.database import Database
from ryd_numerov.model.quantum_defect import QuantumDefect
from ryd_numerov.units import ureg

if TYPE_CHECKING:
    from ryd_numerov.units import NDArray


logger = logging.getLogger(__name__)


class ModelPotential:
    """Model potential parameters for an atomic species and angular momentum.

    Attributes:
        species: Atomic species
        n: Principal quantum number
        l: Orbital angular momentum quantum number
        s: Spin quantum number
        j: Total angular momentum quantum number
        quantum_defect: QuantumDefect object for the atomic species and quantum numbers.
        ac: Polarizability parameter in atomic units.
        Z: Nuclear charge.
        a1: Model potential parameter a1 in atomic units.
        a2: Model potential parameter a2 in atomic units.
        a3: Model potential parameter a3 in atomic units.
        a4: Model potential parameter a4 in atomic units.
        rc: Core radius parameter in atomic units.
        database: Database instance, where the model potential parameters are stored.


    """

    def __init__(
        self,
        species: str,
        n: int,
        l: int,
        s: float,
        j: float,
        quantum_defect: QuantumDefect,
        database: Optional["Database"] = None,
        *,
        add_spin_orbit: bool = True,
        add_model_potentials: bool = True,
    ) -> None:
        r"""Initialize the model potential.

        Args:
            species: Atomic species
            n: Principal quantum number
            l: Orbital angular momentum quantum number
            s: Spin quantum number
            j: Total angular momentum quantum number
            quantum_defect: QuantumDefect object for the atomic species and quantum numbers.
            database: Database instance, where the model potential parameters are stored
              If None, use the global database instance.
            add_spin_orbit: Whether to include the spin-orbit coupling potential in the total physical potential.
            add_model_potentials: Whether to include the model potentials
              (see calc_potential_core and calc_potential_core_polarization)

        """
        self.species = species
        self.n = n
        self.l = l
        self.s = s
        self.j = j
        self.quantum_defect = quantum_defect

        if database is None:
            database = Database.get_global_instance()
        self.database = database

        self.ac, self.Z, self.a1, self.a2, self.a3, self.a4, self.rc = database.get_model_potential_parameters(
            self.species, self.l
        )

        self.add_spin_orbit = add_spin_orbit
        self.add_model_potentials = add_model_potentials

    @property
    def xc(self) -> float:
        """Core radius parameter in dimensionless units."""
        return self.rc

    def calc_potential_core(self, x: "NDArray") -> "NDArray":
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
        if not self.add_model_potentials:
            return -1 / x
        exp_a1 = np.exp(-self.a1 * x)
        exp_a2 = np.exp(-self.a2 * x)
        z_nl: NDArray = 1 + (self.Z - 1) * exp_a1 - x * (self.a3 + self.a4 * x) * exp_a2
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
        if self.ac == 0 or not self.add_model_potentials:
            return np.zeros_like(x)
        x2: NDArray = x * x
        x4: NDArray = x2 * x2
        x6: NDArray = x4 * x2
        exp_x6 = np.exp(-(x6 / self.xc**6))
        v_p: NDArray = -self.ac / (2 * x4) * (1 - exp_x6)
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
        if x[0] < self.xc:
            v_so *= x > self.xc
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
        return (1 / self.quantum_defect.mu) * self.l * (self.l + 1) / (2 * x2)

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
        return (1 / self.quantum_defect.mu) * (3 / 32) / x2

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
        v_tot = (
            self.calc_potential_core(x) + self.calc_potential_core_polarization(x) + self.calc_potential_centrifugal(x)
        )
        if self.add_spin_orbit:
            v_tot += self.calc_potential_spin_orbit(x)
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
            energy = self.quantum_defect.energy
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
