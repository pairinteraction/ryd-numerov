"""Interface to quantum defects SQL database containing model potential and Rydberg-Ritz parameters.

This module provides classes to access and query the quantum defects SQLite database
containing model potential parameters and Rydberg-Ritz coefficients for various atomic species.
"""

import logging
from dataclasses import dataclass

from ryd_numerov.units import ureg

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
class GroundState:
    """Ground state parameters for an atomic species."""

    species: str
    """Atomic species."""
    configuration: str
    """Electron configuration in noble gas notation."""
    n: int
    """Principal quantum number."""
    l: int
    """Orbital angular momentum quantum number."""
    s: float
    """Spin quantum number."""
    j: float
    """Total angular momentum quantum number."""
    m: float
    """Magnetic quantum number."""
    ionization_energy: float
    """Ionization energy in GHz."""

    def get_ionization_energy(self, unit: str = "hartree") -> float:
        """Return the ionization energy in the desired unit.

        Args:
            unit: Desired unit for the ionization energy. Default is atomic units "hartree".

        Returns:
            Ionization energy in the desired unit.

        """
        return ureg.Quantity(self.ionization_energy, "GHz").to(unit, "spectroscopy").magnitude  # type: ignore [no-any-return]  # pint typing .to(unit)

    def is_allowed_shell(self, n: int, l: int) -> bool:
        """Check if the quantum numbers describe a allowed shell (i.e. are above the ground state).

        Args:
            n: Principal quantum number
            l: Orbital angular momentum quantum number

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
        state_id = SORTED_SHELLS.index((n, l))
        return state_id >= gs_id
