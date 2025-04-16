"""Interface to quantum defects SQL database containing model potential and Rydberg-Ritz parameters.

This module provides classes to access and query the quantum defects SQLite database
containing model potential parameters and Rydberg-Ritz coefficients for various atomic species.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ryd_numerov.model.database import Database
from ryd_numerov.units import ureg

logger = logging.getLogger(__name__)


@dataclass
class QuantenDefect:
    """Rydberg-Ritz parameters for an atomic species and quantum numbers."""

    species: str
    """Atomic species."""
    l: int
    """Angular momentum quantum number."""
    j: float
    """Total angular momentum quantum number."""
    d0: float
    """Zeroth-order quantum defect."""
    d2: float
    """Second-order quantum defect."""
    d4: float
    """Fourth-order quantum defect."""
    d6: float
    """Sixth-order quantum defect."""
    d8: float
    """Eighth-order quantum defect."""
    Ry: float
    """Rydberg constant in cm^{-1}"""
    Ry_inf: float = ureg.Quantity(1, "rydberg_constant").to("1/cm").magnitude
    """Rydberg constant in cm^{-1} for infinite nuclear mass."""

    @classmethod
    def from_database(
        cls,
        species: str,
        l: int,
        j: float,
        database: Optional["Database"] = None,
    ) -> "QuantenDefect":
        """Create an instance by taking the parameters from the database.

        Args:
            species: Atomic species
            l: Angular momentum quantum number
            j: Total angular momentum quantum number
            database: Database instance. If None, use the global database instance.

        """
        if database is None:
            database = Database.get_global_instance()
        d0, d2, d4, d6, d8, Ry = database.get_quanten_defect_parameters(species, l, j)  # noqa: N806
        return cls(species, l, j, d0, d2, d4, d6, d8, Ry)

    @property
    def mu(self) -> float:
        r"""Return the reduced mass in atomic units, i.e. return m_{Core} / (m_{Core} + m_e).

        To get the reduced mass in atomic units, we use the species dependent Rydberg constant

        .. math::
            R_{m_{Core}} / R_{\infty} = \frac{m_{Core}}{m_{Core} + m_e}

        """
        return self.Ry / self.Ry_inf

    def get_n_star(self, n: int) -> float:
        """Return the effective principal quantum number.

        Args:
            n: Principal quantum number

        Returns:
            Effective principal quantum number.

        """
        delta_nlj = self.d0 + self.d2 / (n - self.d0) ** 2 + self.d4 / (n - self.d0) ** 4 + self.d6 / (n - self.d0) ** 6
        return n - delta_nlj

    def get_energy(self, n: int) -> float:
        r"""Return the energy of a Rydberg state with principal quantum number n in atomic units.

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

        Args:
            n: Principal quantum number of the state to calculate the energy for.

        Returns:
            Energy of the Rydberg state in atomic units.

        """
        """Return the energy of the state in atomic units.

        Args:
            n: Principal quantum number

        Returns:
            Energy of the state in atomic units.

        """
        return -0.5 * self.mu / self.get_n_star(n) ** 2
