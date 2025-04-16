"""Interface to quantum defects SQL database containing model potential and Rydberg-Ritz parameters.

This module provides classes to access and query the quantum defects SQLite database
containing model potential parameters and Rydberg-Ritz coefficients for various atomic species.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ryd_numerov.model.database import Database

logger = logging.getLogger(__name__)


@dataclass
class ModelPotential:
    """Model potential parameters for an atomic species and angular momentum."""

    species: str
    """Atomic species."""
    l: int
    """Angular momentum quantum number."""
    ac: float
    """Polarizability parameter in atomic units."""
    Z: int
    """Nuclear charge."""
    a1: float
    """Model potential parameter a1 in atomic units."""
    a2: float
    """Model potential parameter a2 in atomic units."""
    a3: float
    """Model potential parameter a3 in atomic units."""
    a4: float
    """Model potential parameter a4 in atomic units."""
    rc: float
    """Core radius parameter in atomic units."""

    @classmethod
    def from_database(
        cls,
        species: str,
        l: int,
        database: Optional["Database"] = None,
    ) -> "ModelPotential":
        """Create an instance by taking the parameters from the database.

        Args:
            species: Atomic species
            l: Angular momentum quantum number
            database: Database instance. If None, use the global database instance.

        """
        if database is None:
            database = Database.get_global_instance()
        ac, Z, a1, a2, a3, a4, rc = database.get_model_potential_parameters(species, l)  # noqa: N806
        return cls(species, l, ac, Z, a1, a2, a3, a4, rc)

    @property
    def xc(self) -> float:
        """Core radius parameter in dimensionless units."""
        return self.rc
