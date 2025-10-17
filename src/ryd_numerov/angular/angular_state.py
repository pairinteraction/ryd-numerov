from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Generic, Self, TypeVar

import numpy as np

from ryd_numerov.angular.angular_ket import AngularKetBase, AngularKetFJ, AngularKetJJ, AngularKetLS

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ryd_numerov.angular.angular_ket import CouplingScheme
    from ryd_numerov.units import OperatorType


logger = logging.getLogger(__name__)


_AngularKet = TypeVar("_AngularKet", bound=AngularKetBase)


class AngularState(Generic[_AngularKet]):
    def __init__(self, coefficients: list[float], kets: list[_AngularKet]) -> None:
        self.coefficients = np.array(coefficients)
        self.kets = kets

        if len(coefficients) != len(kets):
            raise ValueError("Length of coefficients and kets must be the same.")
        if not all(type(ket) is type(kets[0]) for ket in kets):
            raise ValueError("All kets must be of the same type.")
        if len(set(kets)) != len(kets):
            raise ValueError("AngularState initialized with duplicate kets.")
        if abs(self.norm - 1) > 1e-10:
            raise ValueError(f"Coefficients must be normalized, but {coefficients=}, {kets=}.")
        if self.norm > 1:
            self.coefficients /= self.norm

    def __iter__(self) -> Iterator[tuple[float, _AngularKet]]:
        return zip(self.coefficients, self.kets).__iter__()

    def __repr__(self) -> str:
        terms = [f"{coeff}*{ket!r}" for coeff, ket in self]
        return f"{self.__class__.__name__}({', '.join(terms)})"

    def __str__(self) -> str:
        terms = [f"{coeff}*{ket!s}" for coeff, ket in self]
        return f"{', '.join(terms)}"

    @property
    def norm(self) -> float:
        """Return the norm of the state (should be 1)."""
        return np.linalg.norm(self.coefficients)  # type: ignore [return-value]

    def _to_coupling_scheme(self, coupling_scheme: CouplingScheme) -> AngularState[AngularKetBase]:
        """Convert to specified coupling scheme."""
        kets: list[AngularKetBase] = []
        coefficients: list[float] = []
        for coeff, ket in self:
            for scheme_coeff, scheme_ket in ket._to_coupling_scheme(coupling_scheme):  # noqa: SLF001
                if scheme_ket in kets:
                    index = kets.index(scheme_ket)
                    coefficients[index] += coeff * scheme_coeff
                else:
                    kets.append(scheme_ket)
                    coefficients.append(coeff * scheme_coeff)
        return AngularState(coefficients, kets)

    def to_ls(self) -> AngularState[AngularKetLS]:
        """Convert to state in LS coupling."""
        return self._to_coupling_scheme("LS")  # type: ignore [return-value]

    def to_jj(self) -> AngularState[AngularKetJJ]:
        """Convert to state in JJ coupling."""
        return self._to_coupling_scheme("JJ")  # type: ignore [return-value]

    def to_fj(self) -> AngularState[AngularKetFJ]:
        """Convert to state in FJ coupling."""
        return self._to_coupling_scheme("FJ")  # type: ignore [return-value]

    def exp_q(self, q: str) -> float:
        """Calculate the expectation value of a quantum number q.

        Args:
            q: The quantum number to calculate the expectation value for.

        """
        if q not in self.kets[0].spin_quantum_numbers_dict:
            raise ValueError(f"Quantum number {q} not found in kets.")

        qs = np.array([ket.spin_quantum_numbers_dict[q] for ket in self.kets])
        if all(q_val == qs[0] for q_val in qs):
            return qs[0]  # type: ignore [no-any-return]

        return np.sum(self.coefficients * self.coefficients * qs)  # type: ignore [no-any-return]

    def std_q(self, q: str) -> float:
        """Calculate the standard deviation of a quantum number q.

        Args:
            q: The quantum number to calculate the standard deviation for.

        """
        if q not in self.kets[0].spin_quantum_numbers_dict:
            raise ValueError(f"Quantum number {q} not found in kets.")

        qs = np.array([ket.spin_quantum_numbers_dict[q] for ket in self.kets])
        if all(q_val == qs[0] for q_val in qs):
            return 0

        coefficients2 = self.coefficients * self.coefficients
        exp_q = np.sum(coefficients2 * qs)
        exp_q2 = np.sum(coefficients2 * qs * qs)

        if abs(exp_q2 - exp_q**2) < 1e-10:
            return 0.0
        return np.sqrt(exp_q2 - exp_q**2)  # type: ignore [no-any-return]

    def calc_reduced_overlap(self, other: AngularState[AngularKetBase] | AngularKetBase) -> float:
        """Calculate the reduced (ignore any m) overlap <self||other>."""
        if isinstance(other, AngularKetBase):
            other = other.to_state()

        raise NotImplementedError("calc_reduced_overlap is not implemented yet")

    def calc_reduced_matrix_element(
        self: Self, other: AngularState[AngularKetBase] | AngularKetBase, operator: OperatorType, kappa: int
    ) -> float:
        r"""Calculate the reduced angular matrix element.

        This means, calculate the following matrix element:

        .. math::
            <self || \hat{O}^{(\kappa)} || other>

        """
        if isinstance(other, AngularKetBase):
            other = other.to_state()

        raise NotImplementedError("calc_reduced_matrix_element is not implemented yet")
