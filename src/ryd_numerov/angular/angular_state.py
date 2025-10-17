from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from ryd_numerov.angular.angular_ket import AngularKetBase

if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)


_AngularKet = TypeVar("_AngularKet", bound=AngularKetBase)


class AngularState(Generic[_AngularKet]):
    def __init__(self, coefficients: list[float], states: list[_AngularKet]) -> None:
        self.coefficients = np.array(coefficients)
        self.states = states

        if len(coefficients) != len(states):
            raise ValueError("Length of coefficients and states must be the same.")
        if abs(self.norm - 1) > 1e-10:
            raise ValueError(f"Coefficients must be normalized, but {coefficients=}, {states=}.")
        if self.norm > 1:
            self.coefficients /= self.norm

    def __iter__(self) -> Iterator[tuple[float, _AngularKet]]:
        return zip(self.coefficients, self.states).__iter__()

    def __repr__(self) -> str:
        terms = [f"{coeff}*{state!r}" for coeff, state in self]
        return f"{self.__class__.__name__}({', '.join(terms)})"

    def __str__(self) -> str:
        terms = [f"{coeff}*{state!s}" for coeff, state in self]
        return f"{', '.join(terms)}"

    @property
    def norm(self) -> float:
        """Return the norm of the superposition state (should be 1)."""
        return np.linalg.norm(self.coefficients)  # type: ignore [return-value]

    def exp_q(self, q: str) -> float:
        """Calculate the expectation value of a quantum number q.

        Args:
            q: The quantum number to calculate the expectation value for.

        """
        if not all(q in state.spin_quantum_numbers_dict for state in self.states):
            raise ValueError(f"Quantum number {q} not found in all states.")

        qs = np.array([state.spin_quantum_numbers_dict[q] for state in self.states])
        if all(q_val == qs[0] for q_val in qs):
            return qs[0]  # type: ignore [no-any-return]

        return np.sum(self.coefficients * self.coefficients * qs)  # type: ignore [no-any-return]

    def std_q(self, q: str) -> float:
        """Calculate the standard deviation of a quantum number q.

        Args:
            q: The quantum number to calculate the standard deviation for.

        """
        if not all(q in state.spin_quantum_numbers_dict for state in self.states):
            raise ValueError(f"Quantum number {q} not found in all states.")

        qs = np.array([state.spin_quantum_numbers_dict[q] for state in self.states])
        if all(q_val == qs[0] for q_val in qs):
            return 0

        coefficients2 = self.coefficients * self.coefficients
        exp_q = np.sum(coefficients2 * qs)
        exp_q2 = np.sum(coefficients2 * qs * qs)

        if abs(exp_q2 - exp_q**2) < 1e-10:
            return 0.0
        return np.sqrt(exp_q2 - exp_q**2)  # type: ignore [no-any-return]
