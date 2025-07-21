import logging
from typing import TYPE_CHECKING, Literal, Optional, TypeVar, get_args

import numpy as np

if TYPE_CHECKING:
    from ryd_numerov.elements import BaseElement
    from ryd_numerov.units import NDArray


logger = logging.getLogger(__name__)

PotentialType = Literal["coulomb", "model_potential_marinescu_1993", "model_potential_fei_2009"]

XType = TypeVar("XType", "NDArray", float)


class Model:
    """Model to describe the potentials for an atomic state."""

    def __init__(
        self,
        element: "BaseElement",
        l: int,
        potential_type: Optional[PotentialType] = None,
    ) -> None:
        r"""Initialize the model.

        Args:
            element: BaseElement object representing the atomic species.
            l: Orbital angular momentum quantum number
            potential_type: Which potential to use for the model.

        """
        self.element = element
        self.l = l

        if potential_type is None:
            potential_type = self.element.potential_type_default
            if potential_type is None:
                potential_type = "coulomb"
        if potential_type not in get_args(PotentialType):
            raise ValueError(f"Invalid potential type {potential_type}. Must be one of {get_args(PotentialType)}.")
        self.potential_type = potential_type

    def calc_potential_coulomb(self, x: "XType") -> "XType":
        r"""Calculate the Coulomb potential V_Col(x) in atomic units.

        The Coulomb potential is given as

        .. math::
            V_{Col}(x) = -1 / x

        where x = r / a_0.

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate the potential.

        Returns:
            V_Col: The Coulomb potential V_Col(x) in atomic units.

        """
        return -1 / x

    def calc_model_potential_marinescu_1993(self, x: "XType") -> "XType":
        r"""Calculate the model potential by Marinescu et al. (1994) in atomic units.

        The model potential from
        M. Marinescu, Phys. Rev. A 49, 982 (1994), https://journals.aps.org/pra/abstract/10.1103/PhysRevA.49.982
        is given by

        .. math::
            V_{mp,marinescu}(x) = - \frac{Z_{l}}{x} - \frac{\alpha_c}{2x^4} (1 - e^{-x^6/x_c**6})

        where Z_{l} is the effective nuclear charge, :math:`\alpha_c` is the static core dipole polarizability,
        and x_c is the effective core size.

        .. math::
            Z_{l} = 1 + (Z - 1) \exp(-a_1 x) - x (a_3 + a_4 x) \exp(-a_2 x)

        with the nuclear charge Z.

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_{mp,marinescu}: The four parameter potential V_{mp,marinescu}(x) in atomic units.

        """
        parameter_dict = self.element.model_potential_parameter_marinescu_1993
        if len(parameter_dict) == 0:
            raise ValueError("No parametric model potential parameters defined for this element.")
        # default to parameters for the maximum l
        a1, a2, a3, a4 = parameter_dict.get(self.l, parameter_dict[max(parameter_dict.keys())])
        exp_a1 = np.exp(-a1 * x)
        exp_a2 = np.exp(-a2 * x)
        z_nl: XType = 1 + (self.element.Z - 1) * exp_a1 - x * (a3 + a4 * x) * exp_a2
        v_c = -z_nl / x

        alpha_c = self.element.alpha_c_marinescu_1993
        if alpha_c == 0:
            v_p = 0
        else:
            r_c_dict = self.element.r_c_dict_marinescu_1993
            if len(r_c_dict) == 0:
                raise ValueError("No parametric model potential parameters defined for this element.")
            # default to x_c for the maximum l
            x_c = r_c_dict.get(self.l, r_c_dict[max(r_c_dict.keys())])
            x2: XType = x * x
            x4: XType = x2 * x2
            x6: XType = x4 * x2
            exp_x6 = np.exp(-(x6 / x_c**6))
            v_p = -alpha_c / (2 * x4) * (1 - exp_x6)

        return v_c + v_p

    def calc_model_potential_fei_2009(self, x: "XType") -> "XType":
        r"""Calculate the model potential by Fei et al. (2009) in atomic units.

        The four parameter potential from Y. Fei et al., Chin. Phys. B 18, 4349 (2009), https://iopscience.iop.org/article/10.1088/1674-1056/18/10/025
        is given by

        .. math::
            V_{mp,fei}(x) = - \frac{1}{x}
                - \frac{Z-1}{x} \cdot [1 - \alpha + \alpha e^{\beta x^\delta + \gamma x^{2\delta}}]^{-1}

        where Z is the nuclear charge.

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_{mp,fei}: The four parameter potential V_{mp,fei}(x) in atomic units.

        """
        delta, alpha, beta, gamma = self.element.model_potential_parameter_fei_2009
        with np.errstate(over="ignore"):
            denom: XType = 1 - alpha + alpha * np.exp(beta * x**delta + gamma * x ** (2.0 * delta))
            return -1 / x - (self.element.Z - 1) / (x * denom)

    def calc_effective_potential_centrifugal(self, x: "XType") -> "XType":
        r"""Calculate the effective centrifugal potential V_l(x) in atomic units.

        The effective centrifugal potential is given as

        .. math::
            V_l(x) = \frac{l(l+1)}{2x^2}

        where x = r / a_0 and l is the orbital angular momentum quantum number.

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate the potential.

        Returns:
            V_l: The effective centrifugal potential V_l(x) in atomic units.

        """
        x2 = x * x
        return (1 / self.element.reduced_mass_factor) * self.l * (self.l + 1) / (2 * x2)

    def calc_effective_potential_sqrt(self, x: "XType") -> "XType":
        r"""Calculate the effective potential V_sqrt(x) from the sqrt transformation in atomic units.

        The sqrt transformation potential arises from the transformation from the wavefunction u(x) to w(z),
        where x = r / a_0 and w(z) = z^{-1/2} u(x=z^2) = (r/a_0)^{-1/4} sqrt(a_0) r R(r).
        Due to the transformation, an additional term is added to the radial Schrödinger equation,
        which can be written as effective potential V_{sqrt}(x) and is given by

        .. math::
            V_{sqrt}(x) = \frac{3}{32x^2}

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate the potential.

        Returns:
            V_sqrt: The sqrt transformation potential V_sqrt(x) in atomic units.

        """
        x2 = x * x
        return (1 / self.element.reduced_mass_factor) * (3 / 32) / x2

    def calc_total_effective_potential(self, x: "XType") -> "XType":
        r"""Calculate the total effective potential V_eff(x) in atomic units.

        The total effective potential includes all physical and effective potentials:

        .. math::
            V_{eff}(x) = V(x) + V_l(x) + V_{sqrt}(x)

        where V(x) is the physical potential (either Coulomb or a model potential),
        V_l(x) is the effective centrifugal potential,
        and V_{sqrt}(x) is the effective potential from the sqrt transformation.

        Note that we on purpose do not include the spin-orbit potential for several reasons:

        i) The fine structure corrections are important for the energies of the states.
           This includes a) spin-orbit coupling, b) Darwin term, and c) relativistic corrections to the kinetic energy.
           Since we (obviously) can not include the latter two in the model,
           it is only consistent to not include the spin-orbit term either.

        ii) The model potentials are generated without the spin-orbit term,
            since their accuracy is not sufficient to resolve the fine structure corrections at small distances.
            (This can also be seen by running Numerov for low lying states with an energy changed by e.g. 1%,
            which will lead to almost no change in the wavefunction.)

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_eff: The total potential V_eff(x) in atomic units.

        """
        # Note: we do not include the spin-orbit potential, see docstring for details.
        if self.potential_type == "coulomb":
            v = self.calc_potential_coulomb(x)
        elif self.potential_type == "model_potential_marinescu_1993":
            v = self.calc_model_potential_marinescu_1993(x)
        elif self.potential_type == "model_potential_fei_2009":
            v = self.calc_model_potential_fei_2009(x)
        else:
            raise ValueError(f"Invalid potential type {self.potential_type}.")

        v += self.calc_effective_potential_centrifugal(x)
        v += self.calc_effective_potential_sqrt(x)

        return v

    def calc_hydrogen_turning_point_z(self, n: int, l: int) -> float:
        r"""Calculate the classical turning point z_i of the state if it would be a hydrogen atom.

        The hydrogen turning point is defined as the point,
        where for the idealized hydrogen atom the potential equals the energy,
        i.e. V_Col(r_i) + V_l(r_i) = E.
        This is exactly the case at

        .. math::
            r_i = n^2 - n \sqrt{n^2 - l(l + 1)}

        and z_i = sqrt{r_i / a_0}.

        Args:
            n: Principal quantum number of the state.
            l: Orbital angular momentum quantum number of the state.

        Returns:
            z_i: The inner hydrogen turning point z_i in the scaled dimensionless coordinate z_i = sqrt{r_i / a_0}.

        """
        return np.sqrt(n * n - n * np.sqrt(n * n - l * (l + 1)))  # type: ignore [no-any-return]

    def calc_turning_point_z(self, energy_au: float, dz: float = 1e-3) -> float:
        r"""Calculate the classical inner turning point z_i for the given state.

        The classical turning point is defined as the point,
        where the total effective potential of the Rydberg model equals the energy,
        i.e. V_eff(r_i) = E.

        Note: Because we use the total effective potential, even for l=0 the turning point is not at r=0.
        The advantage of this is, that this definition of the turning point should correspond to
        where w(z) should have its last change of sign in the second derivative.

        Args:
            energy_au: The energy, for which to calculate the classical turning point in atomic units.
            dz: The precision of the turning point calculation.

        Returns:
            z_i: The inner turning point z_i in the scaled dimensionless coordinate z_i = sqrt{r_i / a_0}.

        """
        # for a given hydrogen turning point z_hyd, the classical turning point usually lies within z_hyd \pm 5
        # for a given l, the hydrogen turning point is bound by
        # z_lower = z_hyd(n=inf, l)  = \sqrt{l * (l+1) / 2} <= z_hyd(n, l) <= z_hyd(n=l+1, l) = z_upper
        z_lower = np.sqrt(self.l * (self.l + 1) / 2)
        z_upper = self.calc_hydrogen_turning_point_z(n=self.l + 1, l=self.l)

        z_min_orig, z_max_orig = max(z_lower - 5, dz), z_upper + 5
        z_min, z_max = z_min_orig, z_max_orig

        while z_max - z_min > dz:
            z_list = np.linspace(z_min, z_max, 1_000, endpoint=True)
            v_list = self.calc_total_effective_potential(z_list**2) - energy_au

            inds = np.argwhere(np.diff(np.sign(v_list)) < 0).flatten()
            if len(inds) == 0:
                raise ValueError("Effective potential is always above or below the energy, this should not happen!")
            ind = inds[-1]  # take the last index, where a sign change from positive to negative occurs
            # because for some potentials, the potential for small distances gets negative again,
            # but the classical forbidden region was already reached for a larger distance

            z_min = z_list[ind]
            z_max = z_list[ind + 1]

        if z_min == z_min_orig or z_max == z_max_orig:
            logger.warning(
                "The turning point calculation did converge to the original z_min or z_max. "
                "This should not happen and is probably a bug!"
            )

        return z_min + (z_max - z_min) * v_list[ind] / (v_list[ind] - v_list[ind + 1])  # type: ignore [no-any-return]
