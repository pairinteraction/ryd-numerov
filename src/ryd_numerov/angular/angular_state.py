from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, get_args

import numpy as np

from ryd_numerov.angular.angular_ket import (
    AngularKetBase,
    AngularKetFJ,
    AngularKetJJ,
    AngularKetLS,
)
from ryd_numerov.angular.angular_matrix_element import AngularMomentumQuantumNumbers

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ryd_numerov.angular.angular_ket import CouplingScheme
    from ryd_numerov.angular.angular_matrix_element import AngularOperatorType


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
    def coupling_scheme(self) -> CouplingScheme:
        """Return the coupling scheme of the state."""
        return self.kets[0].coupling_scheme

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

    def calc_exp_qn(self, q: AngularMomentumQuantumNumbers) -> float:
        """Calculate the expectation value of a quantum number q.

        Args:
            q: The quantum number to calculate the expectation value for.

        """
        if q not in self.kets[0].spin_quantum_number_names:
            for ket_class in [AngularKetLS, AngularKetJJ, AngularKetFJ]:
                if q in ket_class.spin_quantum_number_names:
                    return self._to_coupling_scheme(ket_class.coupling_scheme).calc_exp_qn(q)

        qs = np.array([ket.get_qn(q) for ket in self.kets])
        if all(q_val == qs[0] for q_val in qs):
            return qs[0]  # type: ignore [no-any-return]

        return np.sum(np.conjugate(self.coefficients) * self.coefficients * qs)  # type: ignore [no-any-return]

    def calc_std_qn(self, q: AngularMomentumQuantumNumbers) -> float:
        """Calculate the standard deviation of a quantum number q.

        Args:
            q: The quantum number to calculate the standard deviation for.

        """
        if q not in self.kets[0].spin_quantum_number_names:
            for ket_class in [AngularKetLS, AngularKetJJ, AngularKetFJ]:
                if q in ket_class.spin_quantum_number_names:
                    return self._to_coupling_scheme(ket_class.coupling_scheme).calc_std_qn(q)

        qs = np.array([ket.get_qn(q) for ket in self.kets])
        if all(q_val == qs[0] for q_val in qs):
            return 0

        coefficients2 = np.conjugate(self.coefficients) * self.coefficients
        exp_q = np.sum(coefficients2 * qs)
        exp_q2 = np.sum(coefficients2 * qs * qs)

        if abs(exp_q2 - exp_q**2) < 1e-10:
            return 0
        return np.sqrt(exp_q2 - exp_q**2)  # type: ignore [no-any-return]

    def calc_reduced_overlap(self, other: AngularState[Any] | AngularKetBase) -> float:
        """Calculate the reduced overlap <self||other> (ignoring the magnetic quantum number m)."""
        if isinstance(other, AngularKetBase):
            other = other.to_state()

        ov = 0
        for coeff1, ket1 in self:
            for coeff2, ket2 in other:
                ov += np.conjugate(coeff1) * coeff2 * ket1.calc_reduced_overlap(ket2)
        return ov

    def calc_reduced_matrix_element(
        self: Self, other: AngularState[Any] | AngularKetBase, operator: AngularOperatorType, kappa: int
    ) -> float:
        r"""Calculate the reduced angular matrix element.

        This means, calculate the following matrix element:

        .. math::
            \left\langle self || \hat{O}^{(\kappa)} || other \right\rangle

        """
        if isinstance(other, AngularKetBase):
            other = other.to_state()
        if (
            operator in get_args(AngularMomentumQuantumNumbers)
            and operator not in self.kets[0].spin_quantum_number_names
        ):
            for ket_class in [AngularKetLS, AngularKetJJ, AngularKetFJ]:
                if operator in ket_class.spin_quantum_number_names:
                    return self._to_coupling_scheme(ket_class.coupling_scheme).calc_reduced_matrix_element(
                        other, operator, kappa
                    )

        if self.coupling_scheme != other.coupling_scheme:
            other = other._to_coupling_scheme(self.coupling_scheme)  # noqa: SLF001

        value = 0
        for coeff1, ket1 in self:
            for coeff2, ket2 in other:
                value += np.conjugate(coeff1) * coeff2 * ket1.calc_reduced_matrix_element(ket2, operator, kappa)
        return value
