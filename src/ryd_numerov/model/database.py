import logging
import sqlite3
from pathlib import Path
from typing import ClassVar, Optional

logger = logging.getLogger(__name__)


class Database:
    """Interface to model potential SQL database."""

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

    def __del__(self) -> None:
        """Close database connection on object deletion."""
        self.conn.close()
