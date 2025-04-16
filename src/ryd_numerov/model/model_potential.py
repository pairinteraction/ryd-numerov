"""Interface to quantum defects SQL database containing model potential and Rydberg-Ritz parameters.

This module provides classes to access and query the quantum defects SQLite database
containing model potential parameters and Rydberg-Ritz coefficients for various atomic species.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelPotentialParameters:
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

    def __post_init__(self) -> None:
        if isinstance(self.rc, str) and self.rc.lower() == "inf":
            self.rc = np.inf

    @property
    def xc(self) -> float:
        """Core radius parameter in dimensionless units."""
        return self.rc
