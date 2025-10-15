from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from ryd_numerov.angular import calc_angular_matrix_element, clebsch_gordan_6j, clebsch_gordan_9j
from ryd_numerov.elements.base_element import BaseElement

if TYPE_CHECKING:
    from ryd_numerov.units import OperatorType

logger = logging.getLogger(__name__)


class SpinStateBase(ABC):
    """Base class for a spin state."""

    i_c: float
    """Nuclear spin."""
    s_c: float
    """Core electron spin (0 for alkali metals, 0.5 for alkaline earth metals)."""
    s_r: float
    """Rydberg electron spin (always 0.5)."""
    l_c: int
    """Core electron orbital angular momentum"""
    l_r: int
    """Rydberg electron orbital angular momentum"""

    f_tot: float
    """Total spin (including nuclear, core electron and rydberg electron contributions)."""

    species: str | None
    """Atomic species, e.g. 'Rb87'.
    Not used for calculations, only for convenience to infert core electron+ spin and nuclear spin."""
    m: float | None
    """Magnetic quantum number.
    If None, only reduced matrix elements can be calculated
    """

    @property
    @abstractmethod
    def spin_quantum_numbers_dict(self) -> dict[str, float | int]:
        """Return the spin quantum numbers (i.e. without the magnetic quantum number) as dictionary."""

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.spin_quantum_numbers_dict.items())
        if self.m is not None:
            args += f", m={self.m}"
        if self.species is not None:
            args += f", species='{self.species}'"
        return f"{self.__class__.__name__}({args})"

    def __str__(self) -> str:
        return self.__repr__()

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if self.s_c not in [0, 0.5]:
            msgs.append(f"Core spin s_c must be 0 or 1/2, but {self.s_c=}")
        if self.s_r != 0.5:
            msgs.append(f"Rydberg electron spin s_r must be 1/2, but {self.s_r=}")

        if self.m is not None and not -self.f_tot <= self.m <= self.f_tot:
            msgs.append(f"m must be between -f_tot and f_tot, but {self.f_tot=}, {self.m=}")

        for msg in msgs:
            logger.error(msg)
        if msgs:
            raise ValueError(f"Invalid quantum numbers for {self!r}")

    def calc_reduced_overlap(self, other: SpinStateBase) -> float:
        """Calculate the reduced (ignore any m) overlap <self||other>."""
        if type(self) is type(other):
            for k, qn1 in self.spin_quantum_numbers_dict.items():
                if qn1 != other.spin_quantum_numbers_dict[k]:
                    return 0.0
            return 1.0

        states = [self, other]

        if any(isinstance(s, SpinStateJJ) for s in states) and any(isinstance(s, SpinStateFJ) for s in states):
            jj = next(s for s in states if isinstance(s, SpinStateJJ))
            fj = next(s for s in states if isinstance(s, SpinStateFJ))
            return clebsch_gordan_6j(fj.i_c, fj.j_c, fj.f_c, fj.j_r, fj.f_tot, jj.j_tot)

        if any(isinstance(s, SpinStateJJ) for s in states) and any(isinstance(s, SpinStateLS) for s in states):
            jj = next(s for s in states if isinstance(s, SpinStateJJ))
            ls = next(s for s in states if isinstance(s, SpinStateLS))
            return clebsch_gordan_9j(ls.s_r, ls.s_c, ls.s_tot, ls.l_r, ls.l_c, ls.l_tot, jj.j_r, jj.j_c, jj.j_tot)

        if any(isinstance(s, SpinStateFJ) for s in states) and any(isinstance(s, SpinStateLS) for s in states):
            fj = next(s for s in states if isinstance(s, SpinStateFJ))
            ls = next(s for s in states if isinstance(s, SpinStateLS))
            fj_as_jj = fj.to_jj()
            ov = 0.0
            for coeff, jj in zip(fj_as_jj.coefficients, fj_as_jj.states):
                ov += coeff * ls.calc_reduced_overlap(jj)
            return ov

        raise NotImplementedError(f"This method is not yet implemented for {self!r} and {other!r}.")

    def calc_reduced_angular_matrix_element(
        self, other: SpinStateBase, operator: OperatorType, k_angular: int
    ) -> float:
        """Calculate the dimensionless angular matrix element.

        This means, calculate the following matrix element:
        <self| T^(k_angular)_q |other>
        """
        raise NotImplementedError(f"This method is not yet implemented for {self!r} and {other!r}.")

    def calc_angular_matrix_element(
        self, other: SpinStateBase, operator: OperatorType, k_angular: int, q: int
    ) -> float:
        """Calculate the dimensionless angular matrix element.

        This means, calculate the following matrix element:
        <self| T^(k_angular)_q |other>
        """
        if self.m is None or other.m is None:
            raise ValueError("m must be set to calculate the angular matrix element.")

        raise NotImplementedError(f"This method is not yet implemented for {self!r} and {other!r}.")
        return calc_angular_matrix_element(
            self.s_tot, self.l, self.j_tot, self.m, other.s_tot, other.l, other.j_tot, other.m, operator, k_angular, q
        )


class SpinStateLS(SpinStateBase):
    """Spin state in LS coupling."""

    def __init__(
        self,
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int = 0,
        s_r: float = 0.5,
        l_r: int | None = None,
        s_tot: float | None = None,
        l_tot: int | None = None,
        j_tot: float | None = None,
        f_tot: float | None = None,
        m: float | None = None,
        species: str | None = None,
    ) -> None:
        """Initialize the Spin state."""
        self.species = species
        if species is not None:
            element = BaseElement.from_species(species)
            if i_c is not None and i_c != element.i_c:
                raise ValueError(f"Nuclear spin i_c={i_c} does not match the element {species} with i_c={element.i_c}.")
            i_c = element.i_c
            s_c = 0.5 * (element.number_valence_electrons - 1)
        if i_c is None:
            raise ValueError("Nuclear spin i_c must be set or a species must be given.")
        self.i_c = i_c

        if s_c is None:
            raise ValueError("Core spin s_c must be set or a species must be given.")
        self.s_c = s_c

        self.l_c = l_c
        self.s_r = s_r
        if l_r is None:
            raise ValueError("Rydberg electron orbital angular momentum l_r must be set.")
        self.l_r = l_r

        self.s_tot = _try_trivial_spin_addition(self.s_c, self.s_r, s_tot, "s_tot")
        self.l_tot = _try_trivial_spin_addition(self.l_c, self.l_r, l_tot, "l_tot")
        self.j_tot = _try_trivial_spin_addition(self.l_tot, self.s_tot, j_tot, "j_tot")
        self.f_tot = _try_trivial_spin_addition(self.j_tot, self.i_c, f_tot, "f_tot")

        self.m = m

        self.sanity_check()

    @property
    def spin_quantum_numbers_dict(self) -> dict[str, float | int]:
        """Return the spin quantum numbers (i.e. without the magnetic quantum number) as dictionary."""
        return {
            "i_c": self.i_c,
            "s_c": self.s_c,
            "l_c": self.l_c,
            "s_r": self.s_r,
            "l_r": self.l_r,
            "s_tot": self.s_tot,
            "l_tot": self.l_tot,
            "j_tot": self.j_tot,
            "f_tot": self.f_tot,
        }

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if not _check_spin_addition_rule(self.l_r, self.l_c, self.l_tot):
            msgs.append(f"{self.l_r=}, {self.l_c=}, {self.l_tot=} don't satisfy spin addition rule.")

        if not _check_spin_addition_rule(self.s_r, self.s_c, self.s_tot):
            msgs.append(f"{self.s_r=}, {self.s_c=}, {self.s_tot=} don't satisfy spin addition rule.")

        if not _check_spin_addition_rule(self.l_tot, self.s_tot, self.j_tot):
            msgs.append(f"{self.l_tot=}, {self.s_tot=}, {self.j_tot=} don't satisfy spin addition rule.")

        if not _check_spin_addition_rule(self.j_tot, self.i_c, self.f_tot):
            msgs.append(f"{self.j_tot=}, {self.i_c=}, {self.f_tot=} don't satisfy spin addition rule.")

        super().sanity_check(msgs)


class SpinStateJJ(SpinStateBase):
    """Spin state in JJ coupling."""

    def __init__(
        self,
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int = 0,
        s_r: float = 0.5,
        l_r: int | None = None,
        j_c: float | None = None,
        j_r: float | None = None,
        j_tot: float | None = None,
        f_tot: float | None = None,
        m: float | None = None,
        species: str | None = None,
    ) -> None:
        """Initialize the Spin state."""
        self.species = species
        if species is not None:
            element = BaseElement.from_species(species)
            if i_c is not None and i_c != element.i_c:
                raise ValueError(f"Nuclear spin i_c={i_c} does not match the element {species} with i_c={element.i_c}.")
            i_c = element.i_c
            s_c = 0.5 * (element.number_valence_electrons - 1)
        if i_c is None:
            raise ValueError("Nuclear spin i_c must be set or a species must be given.")
        self.i_c = i_c

        if s_c is None:
            raise ValueError("Core spin s_c must be set or a species must be given.")
        self.s_c = s_c

        self.l_c = l_c
        self.s_r = s_r
        if l_r is None:
            raise ValueError("Rydberg electron orbital angular momentum l_r must be set.")
        self.l_r = l_r

        self.j_c = _try_trivial_spin_addition(self.l_c, self.s_c, j_c, "j_c")
        self.j_r = _try_trivial_spin_addition(self.l_r, self.s_r, j_r, "j_r")
        self.j_tot = _try_trivial_spin_addition(self.j_c, self.j_r, j_tot, "j_tot")
        self.f_tot = _try_trivial_spin_addition(self.j_tot, self.i_c, f_tot, "f_tot")

        self.m = m

        self.sanity_check()

    @property
    def spin_quantum_numbers_dict(self) -> dict[str, float | int]:
        """Return the spin quantum numbers (i.e. without the magnetic quantum number) as dictionary."""
        return {
            "i_c": self.i_c,
            "s_c": self.s_c,
            "l_c": self.l_c,
            "s_r": self.s_r,
            "l_r": self.l_r,
            "j_c": self.j_c,
            "j_r": self.j_r,
            "j_tot": self.j_tot,
            "f_tot": self.f_tot,
        }

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if not _check_spin_addition_rule(self.l_c, self.s_c, self.j_c):
            msgs.append(f"{self.l_c=}, {self.s_c=}, {self.j_c=} don't satisfy spin addition rule.")

        if not _check_spin_addition_rule(self.l_r, self.s_r, self.j_r):
            msgs.append(f"{self.l_r=}, {self.s_r=}, {self.j_r=} don't satisfy spin addition rule.")

        if not _check_spin_addition_rule(self.j_c, self.j_r, self.j_tot):
            msgs.append(f"{self.j_c=}, {self.j_r=}, {self.j_tot=} don't satisfy spin addition rule.")

        if not _check_spin_addition_rule(self.j_tot, self.i_c, self.f_tot):
            msgs.append(f"{self.j_tot=}, {self.i_c=}, {self.f_tot=} don't satisfy spin addition rule.")

        super().sanity_check(msgs)


class SpinStateFJ(SpinStateBase):
    """Spin state in JJ coupling."""

    def __init__(
        self,
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int = 0,
        s_r: float = 0.5,
        l_r: int | None = None,
        j_c: float | None = None,
        j_r: float | None = None,
        f_c: float | None = None,
        f_tot: float | None = None,
        m: float | None = None,
        species: str | None = None,
    ) -> None:
        """Initialize the Spin state."""
        self.species = species
        if species is not None:
            element = BaseElement.from_species(species)
            if i_c is not None and i_c != element.i_c:
                raise ValueError(f"Nuclear spin i_c={i_c} does not match the element {species} with i_c={element.i_c}.")
            i_c = element.i_c
            s_c = 0.5 * (element.number_valence_electrons - 1)
        if i_c is None:
            raise ValueError("Nuclear spin i_c must be set or a species must be given.")
        self.i_c = i_c

        if s_c is None:
            raise ValueError("Core spin s_c must be set or a species must be given.")
        self.s_c = s_c

        self.l_c = l_c
        self.s_r = s_r
        if l_r is None:
            raise ValueError("Rydberg electron orbital angular momentum l_r must be set.")
        self.l_r = l_r

        self.j_c = _try_trivial_spin_addition(self.l_c, self.s_c, j_c, "j_c")
        self.j_r = _try_trivial_spin_addition(self.l_r, self.s_r, j_r, "j_r")
        self.f_c = _try_trivial_spin_addition(self.j_c, self.i_c, f_c, "f_c")
        self.f_tot = _try_trivial_spin_addition(self.f_c, self.j_r, f_tot, "f_tot")

        self.m = m

        self.sanity_check()

    @property
    def spin_quantum_numbers_dict(self) -> dict[str, float | int]:
        """Return the spin quantum numbers (i.e. without the magnetic quantum number) as dictionary."""
        return {
            "i_c": self.i_c,
            "s_c": self.s_c,
            "l_c": self.l_c,
            "s_r": self.s_r,
            "l_r": self.l_r,
            "j_c": self.j_c,
            "f_c": self.f_c,
            "j_r": self.j_r,
            "f_tot": self.f_tot,
        }

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if not _check_spin_addition_rule(self.l_c, self.s_c, self.j_c):
            msgs.append(f"{self.l_c=}, {self.s_c=}, {self.j_c=} don't satisfy spin addition rule.")

        if not _check_spin_addition_rule(self.l_r, self.s_r, self.j_r):
            msgs.append(f"{self.l_r=}, {self.s_r=}, {self.j_r=} don't satisfy spin addition rule.")

        if not _check_spin_addition_rule(self.j_c, self.i_c, self.f_c):
            msgs.append(f"{self.j_c=}, {self.i_c=}, {self.f_c=} don't satisfy spin addition rule.")

        if not _check_spin_addition_rule(self.f_c, self.j_r, self.f_tot):
            msgs.append(f"{self.f_c=}, {self.j_r=}, {self.f_tot=} don't satisfy spin addition rule.")

        super().sanity_check(msgs)

    def to_jj(self) -> SuperpositionState[SpinStateJJ]:
        """Convert to JJ coupling.

        Note that in general this is a superposition of states.
        """
        states: list[SpinStateJJ] = []
        coefficients: list[float] = []

        for j_tot in np.arange(abs(self.f_tot - self.i_c), self.f_tot + self.i_c + 1):
            jj_state = SpinStateJJ(
                self.i_c, self.s_c, self.l_c, self.s_r, self.l_r, self.j_c, self.j_r, float(j_tot), self.f_tot, self.m
            )
            coeff = self.calc_reduced_overlap(jj_state)
            if coeff != 0:
                states.append(jj_state)
                coefficients.append(coeff)

        return SuperpositionState(coefficients, states)


def _try_trivial_spin_addition(s_1: float, s_2: float, s_tot: float | None, name: str) -> float:
    """Try to determine s_tot from s_1 and s_2 if it is not given.

    If s_tot is None and cannot be uniquely determined from s_1 and s_2, raise an error.
    Otherwise return s_tot or the trivial sum s_1 + s_2.
    """
    if s_tot is None:
        if s_1 != 0 and s_2 != 0:
            msg = f"{name} must be set if both parts ({s_1=} and {s_2=}) are non-zero."
            raise ValueError(msg)
        s_tot = s_1 + s_2
    return s_tot


def _check_spin_addition_rule(s_1: float, s_2: float, s_tot: float) -> bool:
    """Check if the spin addition rule is satisfied.

    This means check the following conditions:
    - |s_1 - s_2| <= s_tot <= s_1 + s_2
    - s_1 + s_2 + s_tot is an integer
    """
    return abs(s_1 - s_2) <= s_tot <= s_1 + s_2 and (s_1 + s_2 + s_tot) % 1 == 0


SpinState = TypeVar("SpinState", bound=SpinStateBase)


class SuperpositionState(Generic[SpinState]):
    def __init__(self, coefficients: list[float], states: list[SpinState]) -> None:
        if len(coefficients) != len(states):
            raise ValueError("Length of coefficients and states must be the same.")
        if abs(np.linalg.norm(coefficients) - 1) > 1e-6:
            raise ValueError("Coefficients must be normalized.")

        self.coefficients = coefficients
        self.states = states
