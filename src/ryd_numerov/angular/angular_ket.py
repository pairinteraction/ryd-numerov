from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Self, TypeVar

import numpy as np

from ryd_numerov.angular.angular_matrix_element import (
    calc_prefactor_of_operator_in_coupled_scheme,
    calc_reduced_spherical_matrix_element,
    calc_reduced_spin_matrix_element,
)
from ryd_numerov.angular.utils import calc_wigner_3j, clebsch_gordan_6j, clebsch_gordan_9j
from ryd_numerov.elements import BaseElement

if TYPE_CHECKING:
    from ryd_numerov.angular.angular_state import AngularState
    from ryd_numerov.units import OperatorType

logger = logging.getLogger(__name__)


class InvalidQuantumNumbersError(ValueError):
    def __init__(self, ket: AngularKetBase, msg: str = "") -> None:
        _msg = f"Invalid quantum numbers for {ket!r}"
        if len(msg) > 0:
            _msg += f"\n  {msg}"
        super().__init__(_msg)


class AngularKetBase(ABC):
    """Base class for a angular ket (i.e. a simple canonical spin ketstate)."""

    # We use __slots__ to prevent dynamic attributes and make the objects immutable after initialization
    __slots__ = ("i_c", "s_c", "l_c", "s_r", "l_r", "f_tot", "m", "_initialized")

    _spin_quantum_number_names: ClassVar[list[str]]
    """Names of all well defined spin quantum numbers (without the magnetic quantum number m) in this class."""

    _coupled_quantum_numbers: ClassVar[dict[str, tuple[str, str]]]
    """Mapping of coupled quantum numbers to their constituent quantum numbers."""

    i_c: float
    """Nuclear spin quantum number."""
    s_c: float
    """Core electron spin quantum number (0 for alkali atoms, 0.5 for alkaline earth atoms)."""
    l_c: int
    """Core electron orbital quantum number (usually 0)."""
    s_r: float
    """Rydberg electron spin quantum number (always 0.5)."""
    l_r: int
    """Rydberg electron orbital quantum number."""

    f_tot: float
    """Total atom angular quantum number (including nuclear, core electron and rydberg electron contributions)."""
    m: float | None
    """Magnetic quantum number, which is the projection of `f_tot` onto the quantization axis.
    If None, only reduced matrix elements can be calculated
    """

    def __init__(
        self,
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int = 0,
        s_r: float = 0.5,
        l_r: int | None = None,
        f_tot: float | None = None,  # noqa: ARG002
        m: float | None = None,
        species: str | None = None,
    ) -> None:
        """Initialize the Spin ket.

        species:
        Atomic species, e.g. 'Rb87'.
        Not used for calculation, only for convenience to infer the core electron spin and nuclear spin quantum numbers.
        """
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

        # f_tot will be set in the subclasses
        self.m = m

    def _post_init(self) -> None:
        self._initialized = True

        self.sanity_check()

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if self.s_c not in [0, 0.5]:
            msgs.append(f"Core spin s_c must be 0 or 1/2, but {self.s_c=}")
        if self.s_r != 0.5:
            msgs.append(f"Rydberg electron spin s_r must be 1/2, but {self.s_r=}")

        if self.m is not None and not -self.f_tot <= self.m <= self.f_tot:
            msgs.append(f"m must be between -f_tot and f_tot, but {self.f_tot=}, {self.m=}")

        if msgs:
            msg = "\n  ".join(msgs)
            raise InvalidQuantumNumbersError(self, msg)

    def __setattr__(self, key: str, value: object) -> None:
        # We use this custom __setattr__ to make the objects immutable after initialization
        if getattr(self, "_initialized", False):
            raise AttributeError(
                f"Cannot modify attributes of immutable {self.__class__.__name__} objects after initialization."
            )
        super().__setattr__(key, value)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.spin_quantum_numbers_dict.items())
        if self.m is not None:
            args += f", m={self.m}"
        return f"{self.__class__.__name__}({args})"

    def __str__(self) -> str:
        return self.__repr__().replace("AngularKet", "")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AngularKetBase):
            raise NotImplementedError(f"Cannot compare {self!r} with {other!r}.")
        if type(self) is not type(other):
            return False
        if self.m != other.m:
            return False
        return all(self.get_qn(k) == other.get_qn(k) for k in self.spin_quantum_numbers_dict)

    def __hash__(self) -> int:
        return hash(
            (
                tuple((k, v) for k, v in self.spin_quantum_numbers_dict.items()),
                self.m,
            )
        )

    @cached_property
    def spin_quantum_numbers_dict(self) -> dict[str, float | int]:
        """Return the spin quantum numbers (i.e. without the magnetic quantum number) as dictionary."""
        return {k: getattr(self, k) for k in self._spin_quantum_number_names}

    def get_qn(self, qn: str) -> float:
        """Get the value of a quantum number by name."""
        if qn not in self._spin_quantum_number_names:
            raise ValueError(f"Quantum number {qn} not found in {self!r}.")
        return self.spin_quantum_numbers_dict[qn]

    @abstractmethod
    def to_ls(self) -> AngularState[AngularKetLS]: ...

    @abstractmethod
    def to_jj(self) -> AngularState[AngularKetJJ]: ...

    @abstractmethod
    def to_fj(self) -> AngularState[AngularKetFJ]: ...

    def to_state(self: Self) -> AngularState[Self]:
        """Convert the ket to a trivial AngularState with one component."""
        return create_angular_state([1.0], [self])

    def calc_reduced_overlap(self, other: AngularKetBase) -> float:
        """Calculate the reduced (ignore any m) overlap <self||other>.

        If both kets are of the same type (=same coupling scheme), this is just a delta function
        of all spin quantum numbers.
        If the kets are of different types, the overlap is calculated using the corresponding
        Clebsch-Gordan coefficients (/ Wigner-j symbols).
        """
        if type(self) is type(other):
            for k, qn1 in self.spin_quantum_numbers_dict.items():
                if qn1 != other.get_qn(k):
                    return 0.0
            return 1.0

        kets = [self, other]

        # JJ - FJ overlaps
        if any(isinstance(s, AngularKetJJ) for s in kets) and any(isinstance(s, AngularKetFJ) for s in kets):
            jj = next(s for s in kets if isinstance(s, AngularKetJJ))
            fj = next(s for s in kets if isinstance(s, AngularKetFJ))
            return clebsch_gordan_6j(fj.j_c, fj.j_r, jj.j_tot, fj.i_c, fj.f_c, fj.f_tot)

        # JJ - LS overlaps
        if any(isinstance(s, AngularKetJJ) for s in kets) and any(isinstance(s, AngularKetLS) for s in kets):
            jj = next(s for s in kets if isinstance(s, AngularKetJJ))
            ls = next(s for s in kets if isinstance(s, AngularKetLS))
            return clebsch_gordan_9j(ls.s_r, ls.s_c, ls.s_tot, ls.l_r, ls.l_c, ls.l_tot, jj.j_r, jj.j_c, jj.j_tot)

        # FJ - LS overlaps
        if any(isinstance(s, AngularKetFJ) for s in kets) and any(isinstance(s, AngularKetLS) for s in kets):
            fj = next(s for s in kets if isinstance(s, AngularKetFJ))
            ls = next(s for s in kets if isinstance(s, AngularKetLS))
            ov = 0.0
            for coeff, jj_ket in fj.to_jj():
                ov += coeff * ls.calc_reduced_overlap(jj_ket)
            return ov

        raise NotImplementedError(f"This method is not yet implemented for {self!r} and {other!r}.")

    def calc_reduced_matrix_element(self: Self, other: AngularKetBase, operator: OperatorType, kappa: int) -> float:
        r"""Calculate the reduced angular matrix element.

        This means, calculate the following matrix element:

        .. math::
            <self || \hat{O}^{(\kappa)} || other>

        """
        if type(self) is not type(other):
            return self.to_state().calc_reduced_matrix_element(other.to_state(), operator, kappa)

        if operator == "SPHERICAL":
            prefactor = self._calc_prefactor_of_operator_in_coupled_scheme(other, "l_r", kappa)
            complete_reduced_matrix_element = calc_reduced_spherical_matrix_element(self.l_r, other.l_r, kappa)
            return prefactor * complete_reduced_matrix_element

        if operator in self._spin_quantum_number_names:
            prefactor = self._calc_prefactor_of_operator_in_coupled_scheme(other, operator, kappa)
            complete_reduced_matrix_element = calc_reduced_spin_matrix_element(
                self.get_qn(operator), other.get_qn(operator)
            )
            return prefactor * complete_reduced_matrix_element

        raise NotImplementedError("calc_reduced_matrix_element is not implemented yet")

    def calc_matrix_element(self, other: AngularKetBase, operator: OperatorType, kappa: int, q: int) -> float:
        r"""Calculate the dimensionless angular matrix element.

        Use the Wigner-Eckart theorem to calculate the angular matrix element from the reduced matrix element.
        This means, calculate the following matrix element:

        .. math::
            <self| \hat{O}^{(\kappa)}_q |other>
            = <\alpha',f_{tot}',m'| \hat{O}^{(\kappa)}_q |\alpha,f_{tot},m>
            = ... \cdot <\alpha',f_{tot}' || \hat{O}^{(\kappa)} || \alpha,f_{tot}>

        where alpha denotes all other quantum numbers
        and :math:`<\alpha',f_{tot}' || \hat{O}^{(\kappa)} || \alpha,f_{tot}>` is the reduced matrix element
        (see `calc_reduced_matrix_element`).

        Args:
            other: The other AngularKet :math:`|other>`.
            operator: The operator type :math:`\hat{O}_{kq}` for which to calculate the matrix element.
                Can be one of "MAGNETIC", "ELECTRIC", "SPHERICAL".
            kappa: The quantum number :math:`\kappa` of the angular momentum operator.
            q: The quantum number :math:`q` of the angular momentum operator.

        Returns:
            The dimensionless angular matrix element.

        """
        if self.m is None or other.m is None:
            raise ValueError("m must be set to calculate the matrix element.")

        reduced_matrix_element = self.calc_reduced_matrix_element(other, operator, kappa)
        prefactor: float = (-1) ** (other.f_tot - other.m)  # type: ignore [assignment]
        wigner_3j = calc_wigner_3j(other.f_tot, kappa, self.f_tot, -other.m, q, self.m)
        return prefactor * reduced_matrix_element * wigner_3j

    def _calc_prefactor_of_operator_in_coupled_scheme(self, other: AngularKetBase, q: str, kappa: int) -> float:
        """Calculate the prefactor for the complete reduced matrix element.

        This approach is only valid if the operator acts only on one of the well defined quantum numbers.
        """
        if type(self) is not type(other):
            raise ValueError(
                "Both kets must be of the same type to calculate the prefactor of the operator in the coupled scheme."
            )

        if q == "f_tot":
            return 1

        for key, qs in self._coupled_quantum_numbers.items():
            if q in qs:
                q_combined = key
                q2 = qs[1] if qs[0] == q else qs[0]
                break
        else:  # no break
            raise ValueError(f"Quantum number {q} not found in _coupled_quantum_numbers.")

        f1, f2, f_tot = (self.get_qn(q), self.get_qn(q2), self.get_qn(q_combined))
        i1, i2, i_tot = (other.get_qn(q), other.get_qn(q2), other.get_qn(q_combined))
        prefactor = calc_prefactor_of_operator_in_coupled_scheme(f1, f2, f_tot, i1, i2, i_tot, kappa)
        return prefactor * self._calc_prefactor_of_operator_in_coupled_scheme(other, q_combined, kappa)


class AngularKetLS(AngularKetBase):
    """Spin ket in LS coupling."""

    __slots__ = ("s_tot", "l_tot", "j_tot")
    _spin_quantum_number_names: ClassVar = ["i_c", "s_c", "l_c", "s_r", "l_r", "s_tot", "l_tot", "j_tot", "f_tot"]
    _coupled_quantum_numbers: ClassVar = {
        "s_tot": ("s_c", "s_r"),
        "l_tot": ("l_c", "l_r"),
        "j_tot": ("s_tot", "l_tot"),
        "f_tot": ("j_tot", "i_c"),
    }

    s_tot: float
    """Total electron spin quantum number (s_c + s_r)."""
    l_tot: int
    """Total electron orbital quantum number (l_c + l_r)."""
    j_tot: float
    """Total electron angular momentum quantum number (s_tot + l_tot)."""

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
        """Initialize the Spin ket."""
        super().__init__(i_c, s_c, l_c, s_r, l_r, f_tot, m, species)

        self.s_tot = _try_trivial_spin_addition(self.s_c, self.s_r, s_tot, "s_tot")
        self.l_tot = int(_try_trivial_spin_addition(self.l_c, self.l_r, l_tot, "l_tot"))
        self.j_tot = _try_trivial_spin_addition(self.l_tot, self.s_tot, j_tot, "j_tot")
        self.f_tot = _try_trivial_spin_addition(self.j_tot, self.i_c, f_tot, "f_tot")

        super()._post_init()

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

    def to_ls(self) -> AngularState[AngularKetLS]:
        """Convert to state in LS coupling.

        Note, this object is already in LS coupling,
        so this method transforms the ket to a trivial state with one component.
        """
        return self.to_state()

    def to_jj(self) -> AngularState[AngularKetJJ]:
        """Convert to state in JJ coupling.

        Note, a state is in general a superposition of multiple kets in the same coupling scheme.
        """
        kets: list[AngularKetJJ] = []
        coefficients: list[float] = []

        for j_c in np.arange(abs(self.s_c - self.l_c), self.s_c + self.l_c + 1):
            for j_r in np.arange(abs(self.s_r - self.l_r), self.s_r + self.l_r + 1):
                try:
                    jj_ket = AngularKetJJ(
                        i_c=self.i_c,
                        s_c=self.s_c,
                        l_c=self.l_c,
                        s_r=self.s_r,
                        l_r=self.l_r,
                        j_c=float(j_c),
                        j_r=float(j_r),
                        j_tot=self.j_tot,
                        f_tot=self.f_tot,
                        m=self.m,
                    )
                except InvalidQuantumNumbersError:
                    continue
                coeff = self.calc_reduced_overlap(jj_ket)
                if coeff != 0:
                    kets.append(jj_ket)
                    coefficients.append(coeff)

        return create_angular_state(coefficients, kets)

    def to_fj(self) -> AngularState[AngularKetFJ]:
        """Convert to state in FJ coupling.

        Note, a state is in general a superposition of multiple kets in the same coupling scheme.
        """
        jj_state = self.to_jj()
        kets: list[AngularKetFJ] = []
        coefficients: list[float] = []
        for jj_coeff, jj_ket in jj_state:
            for fj_coeff, fj_ket in jj_ket.to_fj():
                if fj_ket in kets:
                    idx = kets.index(fj_ket)
                    coefficients[idx] += jj_coeff * fj_coeff
                else:
                    kets.append(fj_ket)
                    coefficients.append(jj_coeff * fj_coeff)

        return create_angular_state(coefficients, kets)


class AngularKetJJ(AngularKetBase):
    """Spin ket in JJ coupling."""

    __slots__ = ("j_c", "j_r", "j_tot")
    _spin_quantum_number_names: ClassVar = ["i_c", "s_c", "l_c", "s_r", "l_r", "j_c", "j_r", "j_tot", "f_tot"]
    _coupled_quantum_numbers: ClassVar = {
        "j_c": ("s_c", "l_c"),
        "j_r": ("s_r", "l_r"),
        "j_tot": ("j_c", "j_r"),
        "f_tot": ("j_tot", "i_c"),
    }

    j_c: float
    """Total core electron angular quantum number (s_c + l_c)."""
    j_r: float
    """Total rydberg electron angular quantum number (s_r + l_r)."""
    j_tot: float
    """Total electron angular momentum quantum number (j_c + j_r)."""

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
        """Initialize the Spin ket."""
        super().__init__(i_c, s_c, l_c, s_r, l_r, f_tot, m, species)

        self.j_c = _try_trivial_spin_addition(self.l_c, self.s_c, j_c, "j_c")
        self.j_r = _try_trivial_spin_addition(self.l_r, self.s_r, j_r, "j_r")
        self.j_tot = _try_trivial_spin_addition(self.j_c, self.j_r, j_tot, "j_tot")
        self.f_tot = _try_trivial_spin_addition(self.j_tot, self.i_c, f_tot, "f_tot")

        super()._post_init()

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

    def to_ls(self) -> AngularState[AngularKetLS]:
        """Convert to state in LS coupling.

        Note, a state is in general a superposition of multiple kets in the same coupling scheme.
        """
        kets: list[AngularKetLS] = []
        coefficients: list[float] = []

        for s_tot in np.arange(abs(self.s_c - self.s_r), self.s_c + self.s_r + 1):
            for l_tot in np.arange(abs(self.l_c - self.l_r), self.l_c + self.l_r + 1):
                try:
                    ls_ket = AngularKetLS(
                        i_c=self.i_c,
                        s_c=self.s_c,
                        l_c=self.l_c,
                        s_r=self.s_r,
                        l_r=self.l_r,
                        s_tot=float(s_tot),
                        l_tot=int(l_tot),
                        j_tot=self.j_tot,
                        f_tot=self.f_tot,
                        m=self.m,
                    )
                except InvalidQuantumNumbersError:
                    continue
                coeff = self.calc_reduced_overlap(ls_ket)
                if coeff != 0:
                    kets.append(ls_ket)
                    coefficients.append(coeff)

        return create_angular_state(coefficients, kets)

    def to_jj(self) -> AngularState[AngularKetJJ]:
        """Convert to state in JJ coupling.

        Note, this object is already in JJ coupling,
        so this method transforms the ket to a trivial state with one component.
        """
        return self.to_state()

    def to_fj(self) -> AngularState[AngularKetFJ]:
        """Convert to state in FJ coupling.

        Note, a state is in general a superposition of multiple kets in the same coupling scheme.
        """
        kets: list[AngularKetFJ] = []
        coefficients: list[float] = []

        for f_c in np.arange(abs(self.j_c - self.i_c), self.j_c + self.i_c + 1):
            try:
                fj_ket = AngularKetFJ(
                    i_c=self.i_c,
                    s_c=self.s_c,
                    l_c=self.l_c,
                    s_r=self.s_r,
                    l_r=self.l_r,
                    j_c=self.j_c,
                    f_c=float(f_c),
                    j_r=self.j_r,
                    f_tot=self.f_tot,
                    m=self.m,
                )
            except InvalidQuantumNumbersError:
                continue
            coeff = self.calc_reduced_overlap(fj_ket)
            if coeff != 0:
                kets.append(fj_ket)
                coefficients.append(coeff)

        return create_angular_state(coefficients, kets)


class AngularKetFJ(AngularKetBase):
    """Spin ket in FJ coupling."""

    __slots__ = ("j_c", "f_c", "j_r")
    _spin_quantum_number_names: ClassVar = ["i_c", "s_c", "l_c", "s_r", "l_r", "j_c", "f_c", "j_r", "f_tot"]
    _coupled_quantum_numbers: ClassVar = {
        "j_c": ("s_c", "l_c"),
        "f_c": ("j_c", "i_c"),
        "j_r": ("s_r", "l_r"),
        "f_tot": ("f_c", "j_r"),
    }

    j_c: float
    """Total core electron angular quantum number (s_c + l_c)."""
    f_c: float
    """Total core angular quantum number (j_c + i_c)."""
    j_r: float
    """Total rydberg electron angular quantum number (s_r + l_r)."""

    def __init__(
        self,
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int = 0,
        s_r: float = 0.5,
        l_r: int | None = None,
        j_c: float | None = None,
        f_c: float | None = None,
        j_r: float | None = None,
        f_tot: float | None = None,
        m: float | None = None,
        species: str | None = None,
    ) -> None:
        """Initialize the Spin ket."""
        super().__init__(i_c, s_c, l_c, s_r, l_r, f_tot, m, species)

        self.j_c = _try_trivial_spin_addition(self.l_c, self.s_c, j_c, "j_c")
        self.j_r = _try_trivial_spin_addition(self.l_r, self.s_r, j_r, "j_r")
        self.f_c = _try_trivial_spin_addition(self.j_c, self.i_c, f_c, "f_c")
        self.f_tot = _try_trivial_spin_addition(self.f_c, self.j_r, f_tot, "f_tot")

        super()._post_init()

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

    def to_ls(self) -> AngularState[AngularKetLS]:
        """Convert to state in LS coupling.

        Note, a state is in general a superposition of multiple kets in the same coupling scheme.
        """
        jj_state = self.to_jj()
        kets: list[AngularKetLS] = []
        coefficients: list[float] = []
        for jj_coeff, jj_ket in jj_state:
            for ls_coeff, ls_ket in jj_ket.to_ls():
                if ls_ket in kets:
                    idx = kets.index(ls_ket)
                    coefficients[idx] += jj_coeff * ls_coeff
                else:
                    kets.append(ls_ket)
                    coefficients.append(jj_coeff * ls_coeff)
        return create_angular_state(coefficients, kets)

    def to_jj(self) -> AngularState[AngularKetJJ]:
        """Convert to state in JJ coupling.

        Note, a state is in general a superposition of multiple kets in the same coupling scheme.
        """
        kets: list[AngularKetJJ] = []
        coefficients: list[float] = []

        for j_tot in np.arange(abs(self.j_c - self.j_r), self.j_c + self.j_r + 1):
            try:
                jj_ket = AngularKetJJ(
                    i_c=self.i_c,
                    s_c=self.s_c,
                    l_c=self.l_c,
                    s_r=self.s_r,
                    l_r=self.l_r,
                    j_c=self.j_c,
                    j_r=self.j_r,
                    j_tot=float(j_tot),
                    f_tot=self.f_tot,
                    m=self.m,
                )
            except InvalidQuantumNumbersError:
                continue
            coeff = self.calc_reduced_overlap(jj_ket)
            if coeff != 0:
                kets.append(jj_ket)
                coefficients.append(coeff)

        return create_angular_state(coefficients, kets)

    def to_fj(self) -> AngularState[AngularKetFJ]:
        """Convert to state in FJ coupling.

        Note, this object is already in FJ coupling,
        so this method transforms the ket to a trivial state with one component.
        """
        return self.to_state()


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


_AngularKet = TypeVar("_AngularKet", bound=AngularKetBase)


def create_angular_state(
    coefficients: list[float],
    kets: list[_AngularKet],
) -> AngularState[_AngularKet]:
    """Create an AngularState from the given coefficients and kets.

    This is just a convenience function to avoid importing AngularState directly.
    """
    from ryd_numerov.angular.angular_state import AngularState  # noqa: PLC0415

    return AngularState(coefficients, kets)
