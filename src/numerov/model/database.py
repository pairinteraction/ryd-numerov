"""Interface to quantum defects SQL database containing model potential and Rydberg-Ritz parameters.

This module provides classes to access and query the quantum defects SQLite database
containing model potential parameters and Rydberg-Ritz coefficients for various atomic species.
"""

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from numerov.units import ureg

logger = logging.getLogger(__name__)


@dataclass
class ModelPotentialParameters:
    """Model potential parameters for an atomic species and angular momentum.

    Attributes:
        element: Atomic element symbol.
        L: Angular momentum quantum number.
        ac: Polarizability parameter in atomic units.
        Z: Nuclear charge.
        a1: Model potential parameter a1 in atomic units.
        a2: Model potential parameter a2 in atomic units.
        a3: Model potential parameter a3 in atomic units.
        a4: Model potential parameter a4 in atomic units.
        rc: Core radius parameter in atomic units.

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

    def __post_init__(self) -> None:
        if isinstance(self.rc, str) and self.rc.lower() == "inf":
            self.rc = np.inf

    @property
    def xc(self) -> float:
        """Core radius parameter in dimensionless units."""
        return self.rc


@dataclass
class RydbergRitzParameters:
    """Rydberg-Ritz parameters for an atomic species and quantum numbers.

    Attributes:
        element: Atomic element symbol.
        L: Angular momentum quantum number.
        J: Total angular momentum quantum number.
        d0: Zeroth-order quantum defect.
        d2: Second-order quantum defect.
        d4: Fourth-order quantum defect.
        d6: Sixth-order quantum defect.
        d8: Eighth-order quantum defect.
        Ry: Rydberg constant in cm^{-1}
        Ry_inf: Rydberg constant in cm^{-1} for infinite nuclear mass.

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

    Ry_inf: float = ureg.Quantity(1, "rydberg_constant").to("1/cm").magnitude

    @property
    def mu(self) -> float:
        r"""Return the reduced mass in atomic units, i.e. return m_{Core} / (m_{Core} + m_e).

        To get the reduced mass in atomic units, we use the species dependent Rydberg constant

        .. math::
            R_{m_{Core}} / R_{\infty} = \frac{m_{Core}}{m_{Core} + m_e}

        """
        return self.Ry / self.Ry_inf


class QuantumDefectsDatabase:
    """Interface to quantum defects SQL database."""

    def __init__(self, qdd_path: Optional[str] = None) -> None:
        """Initialize database connection.

        Args:
            qdd_path: Optional path to SQLite database file. If None, use the default
                quantum_defects.sql in the same directory as this file.

        """
        if qdd_path is None:
            qdd_path = str(Path(__file__).parent / "quantum_defects.sql")

        self.conn = sqlite3.connect(":memory:")
        with open(qdd_path) as f:
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
