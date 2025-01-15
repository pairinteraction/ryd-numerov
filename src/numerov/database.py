"""Interface to quantum defects SQL database containing model potential and Rydberg-Ritz parameters.

This module provides classes to access and query the quantum defects SQLite database
containing model potential parameters and Rydberg-Ritz coefficients for various atomic species.
"""

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
Ry_inf = 109737.31568525


@dataclass
class ModelPotentialParameters:
    """Model potential parameters for an atomic species and angular momentum.

    Attributes:
        element: Atomic element symbol
        L: Angular momentum quantum number
        ac: Polarizability parameter
        Z: Nuclear charge
        a1: Model potential parameter a1
        a2: Model potential parameter a2
        a3: Model potential parameter a3
        a4: Model potential parameter a4
        rc: Core radius parameter

    """

    element: str
    L: int
    ac: float
    Z: int
    a1: float
    a2: float
    a3: float
    a4: float
    rc: float


@dataclass
class RydbergRitzParameters:
    """Rydberg-Ritz parameters for an atomic species and quantum numbers.

    Attributes:
        element: Atomic element symbol
        L: Angular momentum quantum number
        J: Total angular momentum quantum number
        d0: Zeroth-order quantum defect
        d2: Second-order quantum defect
        d4: Fourth-order quantum defect
        d6: Sixth-order quantum defect
        d8: Eighth-order quantum defect
        Ry: Rydberg constant

    """

    element: str
    L: int
    J: float
    d0: float
    d2: float
    d4: float
    d6: float
    d8: float
    Ry: float

    def get_energy(self, n: int) -> float:
        r"""Return the energy of a Rydberg state with principal quantum number n.

        The effective principal quantum number in quantum defect theory is defined as series expansion

        .. math::
            n^* = n - \\delta_{nlj}

        where

        .. math::
            \\delta_{nlj} = d_0 + \frac{d_2}{(n - d_0)^2} + \frac{d_4}{(n - d_0)^4} + \frac{d_6}{(n - d_0)^6}

        is the quantum defect. The energy of the Rydberg state is then given by

        .. math::
            E_{nlj} = -\frac{1}{2} \frac{Ry}{Ry_\\infty} \frac{1}{n^*}

        Args:
            n: Principal quantum number of the state to calculate the energy for.

        Returns:
            Energy of the Rydberg state in atomic units.

        """
        delta_nlj = self.d0 + self.d2 / (n - self.d0) ** 2 + self.d4 / (n - self.d0) ** 4 + self.d6 / (n - self.d0) ** 6
        nstar = n - delta_nlj
        E_nlj = -(self.Ry / Ry_inf) / nstar**2  # TODO *0.5???
        return E_nlj


class QuantumDefectsDatabase:
    """Interface to quantum defects SQL database."""

    def __init__(self, database_path: Optional[str] = None) -> None:
        """Initialize database connection.

        Args:
            database_path: Optional path to SQLite database file. If None, uses default
                quantum_defects.sql in the same directory as this file.

        """
        if database_path is None:
            database_path = str(Path(__file__).parent / "quantum_defects.sql")

        self.conn = sqlite3.connect(":memory:")
        with open(database_path) as f:
            self.conn.executescript(f.read())

    def get_model_potential(self, element: str, L: int) -> ModelPotentialParameters:
        """Get model potential parameters.

        Args:
            element: Atomic element symbol
            L: Angular momentum quantum number

        Returns:
            ModelPotentialParameters containing the model potential parameters.
            If no exact match is found for L, returns parameters for largest available L.

        Raises:
            ValueError: If no parameters found for element

        """
        # Try exact match first
        cursor = self.conn.execute("SELECT * FROM model_potential WHERE element=? AND L=?", (element, L))
        row = cursor.fetchone()

        if row is not None:
            return ModelPotentialParameters(
                element=row[0], L=row[1], ac=row[2], Z=row[3], a1=row[4], a2=row[5], a3=row[6], a4=row[7], rc=row[8]
            )

        # If no exact match, get all entries for this element
        logger.debug("No model potential parameters found for %s with L=%d, trying largest L", element, L)
        cursor = self.conn.execute("SELECT * FROM model_potential WHERE element=? ORDER BY L DESC", (element,))
        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"No model potential parameters found for {element}")

        return ModelPotentialParameters(
            element=row[0], L=row[1], ac=row[2], Z=row[3], a1=row[4], a2=row[5], a3=row[6], a4=row[7], rc=row[8]
        )

    def get_rydberg_ritz(self, element: str, L: int, J: float) -> RydbergRitzParameters:
        """Get Rydberg-Ritz parameters.

        Args:
            element: Atomic element symbol
            L: Angular momentum quantum number
            J: Total angular momentum quantum number

        Returns:
            RydbergRitzParameters containing the Rydberg-Ritz coefficients.
            If no exact match is found, returns parameters for largest available L and J.

        Raises:
            ValueError: If no parameters found for element

        """
        # Try exact match first
        cursor = self.conn.execute("SELECT * FROM rydberg_ritz WHERE element=? AND L=? AND J=?", (element, L, J))
        row = cursor.fetchone()

        if row is not None:
            return RydbergRitzParameters(
                element=row[0], L=row[1], J=row[2], d0=row[3], d2=row[4], d4=row[5], d6=row[6], d8=row[7], Ry=row[8]
            )

        # If no exact match, get all entries for this element ordered by L and J
        logger.debug(
            "No Rydberg-Ritz parameters found for %s with L=%d and J=%d, trying largest L and J", element, L, J
        )
        cursor = self.conn.execute("SELECT * FROM rydberg_ritz WHERE element=? ORDER BY L DESC, J DESC", (element,))
        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"No Rydberg-Ritz parameters found for {element}")

        return RydbergRitzParameters(
            element=row[0], L=row[1], J=row[2], d0=row[3], d2=row[4], d4=row[5], d6=row[6], d8=row[7], Ry=row[8]
        )

    def __del__(self) -> None:
        """Close database connection on object deletion."""
        self.conn.close()
