import logging
import sqlite3
from pathlib import Path
from typing import ClassVar, Optional

from ryd_numerov.model.ground_state import GroundState

logger = logging.getLogger(__name__)


class Database:
    """Interface to quantum defects SQL database."""

    _global_instance: ClassVar[Optional["Database"]] = None

    def __init__(self, database_file: Optional[str] = None) -> None:
        """Initialize database connection.

        Args:
            database_file: Optional path to SQLite database file. If None, use the default
                database.sql in the same directory as this file.

        """
        if database_file is None:
            database_file = str(Path(__file__).parent / "database.sql")

        self.conn = sqlite3.connect(":memory:")
        with Path(database_file).open() as f:
            self.conn.executescript(f.read())

    @classmethod
    def get_global_instance(cls) -> "Database":
        """Return the global database instance."""
        if cls._global_instance is None:
            cls._global_instance = cls()
        return cls._global_instance

    @classmethod
    def set_global_instance(cls, instance: "Database") -> None:
        """Set the global database instance."""
        cls._global_instance = instance

    def get_model_potential_parameters(
        self, species: str, l: int
    ) -> tuple[float, int, float, float, float, float, float]:
        """Get model potential parameters.

        Args:
            species: Atomic species
            l: Angular momentum quantum number

        Returns:
            The model potential parameters for the given species and l, i.e.:
                ac, Z, a1, a2, a3, a4, rc
                If no match is found for l, returns parameters for largest available l.

        """
        cursor = self.conn.execute("SELECT * FROM model_potential WHERE species=? AND l=?", (species, l))
        row = cursor.fetchone()

        if row is None:
            cursor = self.conn.execute("SELECT * FROM model_potential WHERE species=? ORDER BY l DESC", (species,))
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"No model potential parameters found for {species}")
            logger.debug(
                "No model potential parameters found for %s with l=%d, using values for largest l=%d instead",
                *(species, l, row[1]),
            )

        return float(row[2]), int(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8])

    def get_quantum_defect_parameters(
        self, species: str, l: int, j: float
    ) -> tuple[float, float, float, float, float, float]:
        """Get Rydberg-Ritz parameters.

        Args:
            species: Atomic species
            l: Angular momentum quantum number
            j: Total angular momentum quantum number

        Returns:
            The quantum defect parameters for the given species, l and j, i.e.:
                d0, d2, d4, d6, d8, Ry
                If no match is found for l and j, returns d_i = 0 and the normal Ry.

        """
        cursor = self.conn.execute("SELECT * FROM rydberg_ritz WHERE species=? AND l=? AND j=?", (species, l, j))
        row = cursor.fetchone()
        if row is None:
            cursor = self.conn.execute("SELECT * FROM rydberg_ritz WHERE species=? ORDER BY l DESC, j DESC", (species,))
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"No Rydberg-Ritz parameters found for {species}")
            row = (row[0], row[1], row[2], 0.0, 0.0, 0.0, 0.0, 0.0, row[8])
            logger.debug(
                "No Rydberg-Ritz parameters found for %s with l=%d and j=%d, returning parameters with d_i=0",
                *(species, l, j),
            )
        return row[3], row[4], row[5], row[6], row[7], row[8]

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
