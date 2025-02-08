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


# List of energetically sorted shells
SORTED_SHELLS = [  # (n, l)
    (1, 0),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (4, 0),
    (3, 2),
    (4, 1),
    (5, 0),
    (4, 2),
    (5, 1),
    (6, 0),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (5, 3),
    (6, 2),
    (7, 1),
    (8, 0),
]


@dataclass
class ModelPotentialParameters:
    """Model potential parameters for an atomic species and angular momentum.

    Attributes:
        species: Atomic species.
        l: Angular momentum quantum number.
        ac: Polarizability parameter in atomic units.
        Z: Nuclear charge.
        a1: Model potential parameter a1 in atomic units.
        a2: Model potential parameter a2 in atomic units.
        a3: Model potential parameter a3 in atomic units.
        a4: Model potential parameter a4 in atomic units.
        rc: Core radius parameter in atomic units.

    """

    species: str
    l: int
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
        species: Atomic species.
        l: Angular momentum quantum number.
        j: Total angular momentum quantum number.
        d0: Zeroth-order quantum defect.
        d2: Second-order quantum defect.
        d4: Fourth-order quantum defect.
        d6: Sixth-order quantum defect.
        d8: Eighth-order quantum defect.
        Ry: Rydberg constant in cm^{-1}
        Ry_inf: Rydberg constant in cm^{-1} for infinite nuclear mass.

    """

    species: str
    l: int
    j: float
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


@dataclass
class GroundState:
    """Ground state parameters for an atomic species.

    Attributes:
        species: Atomic species
        configuration: Electron configuration in noble gas notation
        n: Principal quantum number
        l: Orbital angular momentum quantum number
        s: Spin quantum number
        j: Total angular momentum quantum number
        m: Magnetic quantum number

    """

    species: str
    configuration: str
    n: int
    l: int
    s: float
    j: float
    m: float
    ionization_energy: float

    def get_ionization_energy(self, unit: str = "hartree") -> float:
        """Return the ionization energy in the desired unit.

        Args:
            unit: Desired unit for the ionization energy. Default is atomic units "hartree".

        Returns:
            Ionization energy in the desired unit.

        """
        return ureg.Quantity(self.ionization_energy, "GHz").to(unit, "spectroscopy").magnitude

    def is_allowed_shell(self, n: int, l: int) -> bool:
        """Check if the quantum numbers describe a allowed shell (i.e. are above the ground state).

        Args:
            n: Principal quantum number
            l: Orbital angular momentum quantum number
            s: Spin quantum number

        Returns:
            True if the quantum numbers specify a state above the ground state, False otherwise.

        """
        if n > 10:
            return True
        if (n, l) not in SORTED_SHELLS:
            return True
        if self.species in ["Sr_triplet"] and (n, l) == (4, 2):  # Sr_triplet has a special case
            return True
        gs_id = SORTED_SHELLS.index((self.n, self.l))
        id = SORTED_SHELLS.index((n, l))
        return id >= gs_id


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

    def get_model_potential(self, species: str, l: int) -> ModelPotentialParameters:
        """Get model potential parameters.

        Args:
            species: Atomic species
            l: Angular momentum quantum number

        Returns:
            ModelPotentialParameters containing the model potential parameters.
            If no exact match is found for l, returns parameters for largest available l.

        Raises:
            ValueError: If no parameters found for species

        """
        # Try exact match first
        cursor = self.conn.execute("SELECT * FROM model_potential WHERE species=? AND l=?", (species, l))
        row = cursor.fetchone()

        if row is not None:
            return ModelPotentialParameters(
                species=row[0], l=row[1], ac=row[2], Z=row[3], a1=row[4], a2=row[5], a3=row[6], a4=row[7], rc=row[8]
            )

        # If no exact match, get all entries for this species
        logger.debug("No model potential parameters found for %s with l=%d, trying largest l", species, l)
        cursor = self.conn.execute("SELECT * FROM model_potential WHERE species=? ORDER BY l DESC", (species,))
        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"No model potential parameters found for {species}")

        return ModelPotentialParameters(
            species=row[0], l=row[1], ac=row[2], Z=row[3], a1=row[4], a2=row[5], a3=row[6], a4=row[7], rc=row[8]
        )

    def get_rydberg_ritz(self, species: str, l: int, j: float) -> RydbergRitzParameters:
        """Get Rydberg-Ritz parameters.

        Args:
            species: Atomic species
            l: Angular momentum quantum number
            j: Total angular momentum quantum number

        Returns:
            RydbergRitzParameters containing the Rydberg-Ritz coefficients.
            If no exact match is found, returns parameters for largest available l and j.

        Raises:
            ValueError: If no parameters found for species

        """
        # Try exact match first
        cursor = self.conn.execute("SELECT * FROM rydberg_ritz WHERE species=? AND l=? AND j=?", (species, l, j))
        row = cursor.fetchone()

        if row is not None:
            return RydbergRitzParameters(
                species=row[0], l=row[1], j=row[2], d0=row[3], d2=row[4], d4=row[5], d6=row[6], d8=row[7], Ry=row[8]
            )

        # If no exact match, get all entries for this species ordered by l and j
        logger.debug(
            "No Rydberg-Ritz parameters found for %s with l=%d and j=%d, trying largest l and j", species, l, j
        )
        cursor = self.conn.execute("SELECT * FROM rydberg_ritz WHERE species=? ORDER BY l DESC, j DESC", (species,))
        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"No Rydberg-Ritz parameters found for {species}")

        return RydbergRitzParameters(
            species=row[0], l=row[1], j=row[2], d0=row[3], d2=row[4], d4=row[5], d6=row[6], d8=row[7], Ry=row[8]
        )

    def get_ground_state(self, species: str) -> GroundState:
        """Get ground state parameters.

        Args:
            species: Atomic species

        Returns:
            GroundState containing the ground state quantum numbers.

        Raises:
            ValueError: If no parameters found for species

        """
        cursor = self.conn.execute("SELECT * FROM ground_state WHERE species=?", (species,))
        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"No ground state parameters found for {species}")

        return GroundState(
            species=row[0],
            configuration=row[1],
            n=row[2],
            l=row[3],
            s=row[4],
            j=row[5],
            m=row[6],
            ionization_energy=row[7],
        )

    def __del__(self) -> None:
        """Close database connection on object deletion."""
        self.conn.close()
