import logging
from typing import Optional

from ryd_numerov.model.database import Database
from ryd_numerov.units import ureg

logger = logging.getLogger(__name__)


class QuantumDefect:
    """Rydberg-Ritz parameters for an atomic species and quantum numbers.

    Attributes:
        species: Atomic species
        n: Principal quantum number
        l: Orbital angular momentum quantum number
        j: Total angular momentum quantum number
        d0: Zeroth-order quantum defect.
        d2: Second-order quantum defect.
        d4: Fourth-order quantum defect.
        d6: Sixth-order quantum defect.
        d8: Eighth-order quantum defect.
        Ry: Rydberg constant in cm^{-1}
        Ry_inf: Rydberg constant in cm^{-1} for infinite nuclear mass

    """

    Ry_inf: float = ureg.Quantity(1, "rydberg_constant").to("1/cm").magnitude

    def __init__(
        self,
        species: str,
        n: int,
        l: int,
        j: float,
        database: Optional["Database"] = None,
    ) -> None:
        r"""Initialize the model potential.

        Args:
            species: Atomic species
            n: Principal quantum number
            l: Orbital angular momentum quantum number
            j: Total angular momentum quantum number
            database: Database instance, where the model potential parameters are stored
              If None, use the global database instance.

        """
        self.species = species
        self.n = n
        self.l = l
        self.j = j

        if database is None:
            database = Database.get_global_instance()
        self.database = database

        self.d0, self.d2, self.d4, self.d6, self.d8, self.Ry = database.get_quantum_defect_parameters(
            self.species, self.l, self.j
        )

    @property
    def mu(self) -> float:
        r"""The reduced mass in atomic units, i.e. return m_{Core} / (m_{Core} + m_e).

        To get the reduced mass in atomic units, we use the species dependent Rydberg constant

        .. math::
            R_{m_{Core}} / R_{\infty} = \frac{m_{Core}}{m_{Core} + m_e}

        """
        return self.Ry / self.Ry_inf

    @property
    def n_star(self) -> float:
        """The effective principal quantum number."""
        delta_nlj = (
            self.d0
            + self.d2 / (self.n - self.d0) ** 2
            + self.d4 / (self.n - self.d0) ** 4
            + self.d6 / (self.n - self.d0) ** 6
        )
        return self.n - delta_nlj

    @property
    def energy(self) -> float:
        r"""The energy of a Rydberg state with principal quantum number n in atomic units.

        The effective principal quantum number in quantum defect theory is defined as series expansion

        .. math::
            n^* = n - \\delta_{nlj}

        where

        .. math::
            \\delta_{nlj} = d_{0} + \frac{d_{2}}{(n - d_{0})^2}
            + \frac{d_{4}}{(n - d_{0})^4} + \frac{d_{6}}{(n - d_{0})^6}

        is the quantum defect. The energy of the Rydberg state is then given by

        .. math::
            E_{nlj} / E_H = -\frac{1}{2} \frac{Ry}{Ry_\infty} \frac{1}{n^*}

        where :math:`E_H` is the Hartree energy (the atomic unit of energy).
        """
        return -0.5 * self.mu / self.n_star**2
