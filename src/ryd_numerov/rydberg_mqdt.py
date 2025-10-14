from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from ryd_numerov.rydberg import RydbergStateBase
from ryd_numerov.units import BaseQuantities

if TYPE_CHECKING:
    from typing_extensions import Self

    from ryd_numerov.units import PintFloat


logger = logging.getLogger(__name__)


class RydbergStateMQDT(RydbergStateBase):
    def __init__(
        self,
        species: str,
        *,
        nu: float | None = None,
        n: int | None = None,
        l: int | None = None,
        energy_au: float | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            nu: Effective principal quantum number of the rydberg electron,
                which is used to calculate the energy of the state.
            n: Principal quantum number of the rydberg electron.
            l: Orbital angular momentum quantum number of the rydberg electron.
            energy_au: The energy of the Rydberg state in atomic units ("hartree").
                Either `nu` or `energy_au` must be provided.

        """
        self.species = species

        self.n = n
        assert l is not None, "l must be set"
        self.l = l

        self._energy_au: float
        if nu is not None and energy_au is not None:
            raise ValueError("Only one of nu or energy_au can be given.")
        if nu is None and energy_au is None:
            raise ValueError("Either nu or energy_au must be provided.")
        if nu is not None:
            self._energy_au = -0.5 * self.element.reduced_mass_factor / nu**2
        elif energy_au is not None:
            self._energy_au = energy_au

        self.sanity_check()

    def __repr__(self) -> str:
        species, nu, n, l = self.species, self.nu, self.n, self.l
        return f"{self.__class__.__name__}({species}, {nu=}, {n=}, {l=})"

    def __str__(self) -> str:
        return self.get_label("ket")

    def copy(self) -> Self:
        """Create a copy of the Rydberg state."""
        return self.__class__(self.species, nu=self.nu, n=self.n, l=self.l)

    def get_label(self, fmt: Literal["raw", "ket", "bra"]) -> str:
        """Label representing the ket.

        Args:
            fmt: The format of the label, i.e. whether to return the raw label, or the label in ket or bra notation.

        Returns:
            The label of the ket in the given format.

        """
        l_dict = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}

        raw = f"{self.species}:{self.nu:.3f},{l_dict.get(self.l, self.l)}"
        if self.n is not None:
            raw += f"(n={self.n})"

        if fmt == "raw":
            return raw
        if fmt == "ket":
            return f"|{raw}⟩"
        if fmt == "bra":
            return f"⟨{raw}|"
        raise ValueError(f"Unknown fmt {fmt}")

    @cached_property
    def nu(self) -> float:
        """Return the effective quantum number nu = n*."""
        return self.get_n_star()

    def sanity_check(self) -> None:
        """Check that the quantum numbers are valid."""
        msgs: list[str] = []
        nu, n, l = self.nu, self.n, self.l

        if not nu > 0:
            msgs.append(f"nu must be larger than 0, but is {nu=}")

        if n is not None:
            if not isinstance(n, (int, np.integer)):
                msgs.append(f"n must be an integer, but {n=}")
            if not n >= 1:
                msgs.append(f"n must be larger than 0, but is {n=}")
            if not nu <= n:
                logger.warning("n should be larger (or equal) than nu, but n=%d, nu=%f for %r", n, nu, self)

        if not isinstance(l, (int, np.integer)):
            msgs.append(f"l must be an integer, but {l=}, {type(l)=}")
        if n is not None and not 0 <= l <= n - 1:
            msgs.append(f"l must be between 0 and n - 1, but {l=}, {n=}")

        # TODO check is_allowed_shell like in RydbergStateSQDT (we dont know s_tot here ...)

        for msg in msgs:
            logger.error(msg)
        if msgs:
            raise ValueError(f"Invalid Rydberg state {self!r}")

    @overload
    def get_energy(self, unit: None = None) -> PintFloat: ...

    @overload
    def get_energy(self, unit: str) -> float: ...

    def get_energy(self, unit: str | None = None) -> PintFloat | float:
        if unit == "a.u.":
            return self._energy_au
        energy: PintFloat = self._energy_au * BaseQuantities["ENERGY"]
        if unit is None:
            return energy
        return energy.to(unit, "spectroscopy").magnitude
