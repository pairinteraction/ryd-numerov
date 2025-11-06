from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, get_args, overload

from ryd_numerov.angular.angular_matrix_element import (
    AngularMomentumQuantumNumbers,
    AngularOperatorType,
    calc_prefactor_of_operator_in_coupled_scheme,
    calc_reduced_identity_matrix_element,
    calc_reduced_spherical_matrix_element,
    calc_reduced_spin_matrix_element,
)
from ryd_numerov.angular.utils import (
    calc_wigner_3j,
    check_spin_addition_rule,
    clebsch_gordan_6j,
    clebsch_gordan_9j,
    get_possible_quantum_number_values,
    minus_one_pow,
    try_trivial_spin_addition,
)
from ryd_numerov.species import SpeciesObject

if TYPE_CHECKING:
    from ryd_numerov.angular.angular_state import AngularState

logger = logging.getLogger(__name__)

CouplingScheme = Literal["LS", "JJ", "FJ"]


class InvalidQuantumNumbersError(ValueError):
    def __init__(self, ket: AngularKetBase, msg: str = "") -> None:
        _msg = f"Invalid quantum numbers for {ket!r}"
        if len(msg) > 0:
            _msg += f"\n  {msg}"
        super().__init__(_msg)


class AngularKetBase(ABC):
    """Base class for a angular ket (i.e. a simple canonical spin ketstate)."""

    # We use __slots__ to prevent dynamic attributes and make the objects immutable after initialization
    __slots__ = ("i_c", "s_c", "l_c", "s_r", "l_r", "f_tot", "m", "quantum_numbers", "_initialized")

    quantum_number_names: ClassVar[tuple[AngularMomentumQuantumNumbers, ...]]
    """Names of all well defined spin quantum numbers (without the magnetic quantum number m) in this class."""

    quantum_numbers: tuple[float, ...]
    """The quantum numbers corresponding to the quantum_number_names (without the magnetic quantum number m)."""

    coupled_quantum_numbers: ClassVar[
        dict[AngularMomentumQuantumNumbers, tuple[AngularMomentumQuantumNumbers, AngularMomentumQuantumNumbers]]
    ]
    """Mapping of coupled quantum numbers to their constituent quantum numbers."""

    coupling_scheme: CouplingScheme
    """Name of the coupling scheme, e.g. 'LS', 'JJ', or 'FJ'."""

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
        species: str | SpeciesObject | None = None,
    ) -> None:
        """Initialize the Spin ket.

        species:
        Atomic species, e.g. 'Rb87'.
        Not used for calculation, only for convenience to infer the core electron spin and nuclear spin quantum numbers.
        """
        if species is not None:
            if isinstance(species, str):
                species = SpeciesObject.from_name(species)
            if i_c is not None and i_c != species.i_c:
                raise ValueError(f"Nuclear spin i_c={i_c} does not match the species {species} with i_c={species.i_c}.")
            i_c = species.i_c
            if species.i_c is None:
                # use i_c = 0 for species without defined nuclear spin (-> ignore hyperfine)
                i_c = 0.0
            s_c = 0.5 * (species.number_valence_electrons - 1)
        if i_c is None:
            raise ValueError("Nuclear spin i_c must be set or a species must be given.")
        self.i_c = float(i_c)

        if s_c is None:
            raise ValueError("Core spin s_c must be set or a species must be given.")
        self.s_c = float(s_c)

        self.l_c = int(l_c)
        self.s_r = float(s_r)
        if l_r is None:
            raise ValueError("Rydberg electron orbital angular momentum l_r must be set.")
        self.l_r = int(l_r)

        # f_tot will be set in the subclasses
        self.m = None if m is None else float(m)

    def _post_init(self) -> None:
        self.quantum_numbers = tuple(getattr(self, qn) for qn in self.quantum_number_names)

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
        args = ", ".join(f"{qn}={val}" for qn, val in zip(self.quantum_number_names, self.quantum_numbers))
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
        return self.quantum_numbers == other.quantum_numbers

    def __hash__(self) -> int:
        return hash(
            (
                self.quantum_number_names,
                self.quantum_numbers,
                self.m,
            )
        )

    def get_qn(self, qn: AngularMomentumQuantumNumbers) -> float:
        """Get the value of a quantum number by name."""
        if qn not in self.quantum_number_names:
            raise ValueError(f"Quantum number {qn} not found in {self!r}.")
        return getattr(self, qn)  # type: ignore [no-any-return]

    @overload
    def to_state(self, coupling_scheme: Literal["LS"]) -> AngularState[AngularKetLS]: ...

    @overload
    def to_state(self, coupling_scheme: Literal["JJ"]) -> AngularState[AngularKetJJ]: ...

    @overload
    def to_state(self, coupling_scheme: Literal["FJ"]) -> AngularState[AngularKetFJ]: ...

    @overload
    def to_state(self: Self) -> AngularState[Self]: ...

    def to_state(self, coupling_scheme: CouplingScheme | None = None) -> AngularState[Any]:
        """Convert to state in the specified coupling scheme.

        Args:
            coupling_scheme: The coupling scheme to convert to (e.g. "LS", "JJ", "FJ").
                If None, the state will be a trivial state (one component) in the current coupling scheme.

        Returns:
            The angular state in the specified coupling scheme.

        """
        from ryd_numerov.angular.angular_state import AngularState  # noqa: PLC0415

        if coupling_scheme is None or coupling_scheme == self.coupling_scheme:
            return AngularState([1], [self])
        if coupling_scheme == "LS":
            return self._to_state_ls()
        if coupling_scheme == "JJ":
            return self._to_state_jj()
        if coupling_scheme == "FJ":
            return self._to_state_fj()
        raise ValueError(f"Unknown coupling scheme {coupling_scheme!r}.")

    def _to_state_ls(self) -> AngularState[AngularKetLS]:
        """Convert a single ket to state in LS coupling."""
        kets: list[AngularKetLS] = []
        coefficients: list[float] = []

        s_tot_list = get_possible_quantum_number_values(self.s_c, self.s_r, getattr(self, "s_tot", None))
        l_tot_list = get_possible_quantum_number_values(self.l_c, self.l_r, getattr(self, "l_tot", None))
        for s_tot in s_tot_list:
            for l_tot in l_tot_list:
                j_tot_list = get_possible_quantum_number_values(s_tot, l_tot, getattr(self, "j_tot", None))
                for j_tot in j_tot_list:
                    try:
                        ls_ket = AngularKetLS(
                            i_c=self.i_c,
                            s_c=self.s_c,
                            l_c=self.l_c,
                            s_r=self.s_r,
                            l_r=self.l_r,
                            s_tot=s_tot,
                            l_tot=int(l_tot),
                            j_tot=j_tot,
                            f_tot=self.f_tot,
                            m=self.m,
                        )
                    except InvalidQuantumNumbersError:
                        continue
                    coeff = self.calc_reduced_overlap(ls_ket)
                    if coeff != 0:
                        kets.append(ls_ket)
                        coefficients.append(coeff)

        from ryd_numerov.angular.angular_state import AngularState  # noqa: PLC0415

        return AngularState(coefficients, kets)

    def _to_state_jj(self) -> AngularState[AngularKetJJ]:
        """Convert a single ket to state in JJ coupling."""
        kets: list[AngularKetJJ] = []
        coefficients: list[float] = []

        j_c_list = get_possible_quantum_number_values(self.s_c, self.l_c, getattr(self, "j_c", None))
        j_r_list = get_possible_quantum_number_values(self.s_r, self.l_r, getattr(self, "j_r", None))
        for j_c in j_c_list:
            for j_r in j_r_list:
                j_tot_list = get_possible_quantum_number_values(j_c, j_r, getattr(self, "j_tot", None))
                for j_tot in j_tot_list:
                    try:
                        jj_ket = AngularKetJJ(
                            i_c=self.i_c,
                            s_c=self.s_c,
                            l_c=self.l_c,
                            s_r=self.s_r,
                            l_r=self.l_r,
                            j_c=j_c,
                            j_r=j_r,
                            j_tot=j_tot,
                            f_tot=self.f_tot,
                            m=self.m,
                        )
                    except InvalidQuantumNumbersError:
                        continue
                    coeff = self.calc_reduced_overlap(jj_ket)
                    if coeff != 0:
                        kets.append(jj_ket)
                        coefficients.append(coeff)

        from ryd_numerov.angular.angular_state import AngularState  # noqa: PLC0415

        return AngularState(coefficients, kets)

    def _to_state_fj(self) -> AngularState[AngularKetFJ]:
        """Convert a single ket to state in FJ coupling."""
        kets: list[AngularKetFJ] = []
        coefficients: list[float] = []

        j_c_list = get_possible_quantum_number_values(self.s_c, self.l_c, getattr(self, "j_c", None))
        j_r_list = get_possible_quantum_number_values(self.s_r, self.l_r, getattr(self, "j_r", None))
        for j_c in j_c_list:
            f_c_list = get_possible_quantum_number_values(j_c, self.i_c, getattr(self, "f_c", None))
            for f_c in f_c_list:
                for j_r in j_r_list:
                    try:
                        fj_ket = AngularKetFJ(
                            i_c=self.i_c,
                            s_c=self.s_c,
                            l_c=self.l_c,
                            s_r=self.s_r,
                            l_r=self.l_r,
                            j_c=j_c,
                            f_c=f_c,
                            j_r=j_r,
                            f_tot=self.f_tot,
                            m=self.m,
                        )
                    except InvalidQuantumNumbersError:
                        continue
                    coeff = self.calc_reduced_overlap(fj_ket)
                    if coeff != 0:
                        kets.append(fj_ket)
                        coefficients.append(coeff)

        from ryd_numerov.angular.angular_state import AngularState  # noqa: PLC0415

        return AngularState(coefficients, kets)

    def calc_reduced_overlap(self, other: AngularKetBase) -> float:
        """Calculate the reduced overlap <self||other> (ignoring the magnetic quantum number m).

        If both kets are of the same type (=same coupling scheme), this is just a delta function
        of all spin quantum numbers.
        If the kets are of different types, the overlap is calculated using the corresponding
        Clebsch-Gordan coefficients (/ Wigner-j symbols).
        """
        for q in set(self.quantum_number_names) & set(other.quantum_number_names):
            if self.get_qn(q) != other.get_qn(q):
                return 0

        if type(self) is type(other):
            return 1

        kets = [self, other]

        # JJ - FJ overlaps
        if any(isinstance(s, AngularKetJJ) for s in kets) and any(isinstance(s, AngularKetFJ) for s in kets):
            jj = next(s for s in kets if isinstance(s, AngularKetJJ))
            fj = next(s for s in kets if isinstance(s, AngularKetFJ))
            return clebsch_gordan_6j(fj.j_r, fj.j_c, fj.i_c, jj.j_tot, fj.f_c, fj.f_tot)

        # JJ - LS overlaps
        if any(isinstance(s, AngularKetJJ) for s in kets) and any(isinstance(s, AngularKetLS) for s in kets):
            jj = next(s for s in kets if isinstance(s, AngularKetJJ))
            ls = next(s for s in kets if isinstance(s, AngularKetLS))
            # NOTE: it matters, whether you first put all 3 l's and then all 3 s's or the other way round
            # (see symmetry properties of 9j symbol)
            # this convention is used, such that all matrix elements work out correctly, no matter in which
            # coupling scheme they are calculated
            return clebsch_gordan_9j(ls.l_r, ls.l_c, ls.l_tot, ls.s_r, ls.s_c, ls.s_tot, jj.j_r, jj.j_c, jj.j_tot)

        # FJ - LS overlaps
        if any(isinstance(s, AngularKetFJ) for s in kets) and any(isinstance(s, AngularKetLS) for s in kets):
            fj = next(s for s in kets if isinstance(s, AngularKetFJ))
            ls = next(s for s in kets if isinstance(s, AngularKetLS))
            ov: float = 0
            for coeff, jj_ket in fj.to_state("JJ"):
                ov += coeff * ls.calc_reduced_overlap(jj_ket)
            return ov

        raise NotImplementedError(f"This method is not yet implemented for {self!r} and {other!r}.")

    def calc_reduced_matrix_element(  # noqa: C901
        self: Self, other: AngularKetBase, operator: AngularOperatorType, kappa: int
    ) -> float:
        r"""Calculate the reduced angular matrix element.

        We follow equation (7.1.7) from Edmonds 1985 "Angular Momentum in Quantum Mechanics".
        This means, calculate the following matrix element:

        .. math::
            \left\langle self || \hat{O}^{(\kappa)} || other \right\rangle

        """
        if operator not in get_args(AngularOperatorType):
            raise NotImplementedError(f"calc_reduced_matrix_element is not implemented for operator {operator}.")

        if type(self) is not type(other):
            return self.to_state().calc_reduced_matrix_element(other.to_state(), operator, kappa)
        if operator in get_args(AngularMomentumQuantumNumbers) and operator not in self.quantum_number_names:
            return self.to_state().calc_reduced_matrix_element(other.to_state(), operator, kappa)

        qn_name: AngularMomentumQuantumNumbers
        if operator == "spherical":
            qn_name = "l_r"
            complete_reduced_matrix_element = calc_reduced_spherical_matrix_element(self.l_r, other.l_r, kappa)
        elif operator in self.quantum_number_names:
            if not kappa == 1:
                raise ValueError("Only kappa=1 is supported for spin operators.")
            qn_name = operator  # type: ignore [assignment]
            complete_reduced_matrix_element = calc_reduced_spin_matrix_element(
                self.get_qn(qn_name), other.get_qn(qn_name)
            )
        elif operator.startswith("identity_"):
            if not kappa == 0:
                raise ValueError("Only kappa=0 is supported for identity operator.")
            qn_name = operator.replace("identity_", "")  # type: ignore [assignment]
            complete_reduced_matrix_element = calc_reduced_identity_matrix_element(
                self.get_qn(qn_name), other.get_qn(qn_name)
            )
        else:
            raise NotImplementedError(f"calc_reduced_matrix_element is not implemented for operator {operator}.")

        if complete_reduced_matrix_element == 0:
            return 0
        if self._kronecker_delta_non_involved_spins(other, qn_name) == 0:
            return 0
        prefactor = self._calc_prefactor_of_operator_in_coupled_scheme(other, qn_name, kappa)
        return prefactor * complete_reduced_matrix_element

    def calc_matrix_element(self, other: AngularKetBase, operator: AngularOperatorType, kappa: int, q: int) -> float:
        r"""Calculate the dimensionless angular matrix element.

        Use the Wigner-Eckart theorem to calculate the angular matrix element from the reduced matrix element.
        We stick to the convention from Edmonds 1985 "Angular Momentum in Quantum Mechanics", see equation (5.4.1).
        This means, calculate the following matrix element:

        .. math::
            \left\langle self | \hat{O}^{(\kappa)}_q | other \right\rangle
            = <\alpha',f_{tot}',m'| \hat{O}^{(\kappa)}_q |\alpha,f_{tot},m>
            = (-1)^{(f_{tot} - m)} \cdot \mathrm{Wigner3j}(f_{tot}', \kappa, f_{tot}, -m', q, m)
                \cdot <\alpha',f_{tot}' || \hat{O}^{(\kappa)} || \alpha,f_{tot}>

        where alpha denotes all other quantum numbers
        and :math:`<\alpha',f_{tot}' || \hat{O}^{(\kappa)} || \alpha,f_{tot}>` is the reduced matrix element
        (see `calc_reduced_matrix_element`).

        Args:
            other: The other AngularKet :math:`|other>`.
            operator: The operator type :math:`\hat{O}_{kq}` for which to calculate the matrix element.
                E.g. 'spherical', 's_tot', 'l_r', etc.
            kappa: The rank :math:`\kappa` of the angular momentum operator.
            q: The component :math:`q` of the angular momentum operator.

        Returns:
            The dimensionless angular matrix element.

        """
        if self.m is None or other.m is None:
            raise ValueError("m must be set to calculate the matrix element.")

        prefactor = self._calc_wigner_eckart_prefactor(other, kappa, q)
        reduced_matrix_element = self.calc_reduced_matrix_element(other, operator, kappa)
        return prefactor * reduced_matrix_element

    def _calc_wigner_eckart_prefactor(self, other: AngularKetBase, kappa: int, q: int) -> float:
        assert self.m is not None and other.m is not None, "m must be set to calculate the Wigner-Eckart prefactor."  # noqa: PT018
        return minus_one_pow(self.f_tot - self.m) * calc_wigner_3j(self.f_tot, kappa, other.f_tot, -self.m, q, other.m)

    def _kronecker_delta_non_involved_spins(self, other: AngularKetBase, qn: AngularMomentumQuantumNumbers) -> int:
        """Calculate the Kronecker delta for non involved angular momentum quantum numbers.

        This means return 0 if any of the quantum numbers,
        that are not qn or a coupled quantum number resulting from qn differ between self and other.
        """
        if qn not in self.quantum_number_names:
            raise ValueError(f"Quantum number {qn} is not a valid angular momentum quantum number for {self!r}.")

        resulting_qns = {qn}
        last_qn = qn
        while last_qn != "f_tot":
            for key, qs in self.coupled_quantum_numbers.items():
                if last_qn in qs:
                    resulting_qns.add(key)
                    last_qn = key
                    break
            else:
                raise ValueError(
                    f"_kronecker_delta_non_involved_spins: {last_qn} not found in coupled_quantum_numbers."
                )

        non_involved_qns = set(self.quantum_number_names) - resulting_qns
        for _qn in non_involved_qns:
            if self.get_qn(_qn) != other.get_qn(_qn):
                return 0
        return 1

    def _calc_prefactor_of_operator_in_coupled_scheme(
        self, other: AngularKetBase, qn: AngularMomentumQuantumNumbers, kappa: int
    ) -> float:
        """Calculate the prefactor for the complete reduced matrix element.

        This approach is only valid if the operator acts only on one of the well defined quantum numbers.
        """
        if type(self) is not type(other):
            raise ValueError(
                "Both kets must be of the same type to calculate the prefactor of the operator in the coupled scheme."
            )

        if qn == "f_tot":
            return 1

        for key, qs in self.coupled_quantum_numbers.items():
            if qn not in qs:
                continue
            qn_combined = key
            # NOTE: the order does actually matter for the sign of some matrix elements
            # we use this to convention to stay consistent with the old pairinteraction database signs
            qn2, qn1 = qs
            operator_acts_on: Literal["first", "second"] = "first" if qn == qn1 else "second"
            break
        else:  # no break
            raise ValueError(f"Quantum number {qn} not found in coupled_quantum_numbers.")

        f1, f2, f_tot = (self.get_qn(qn1), self.get_qn(qn2), self.get_qn(qn_combined))
        i1, i2, i_tot = (other.get_qn(qn1), other.get_qn(qn2), other.get_qn(qn_combined))

        if (operator_acts_on == "first" and f2 != i2) or (operator_acts_on == "second" and f1 != i1):
            return 0
        prefactor = calc_prefactor_of_operator_in_coupled_scheme(f1, f2, f_tot, i1, i2, i_tot, kappa, operator_acts_on)
        return prefactor * self._calc_prefactor_of_operator_in_coupled_scheme(other, qn_combined, kappa)


class AngularKetLS(AngularKetBase):
    """Spin ket in LS coupling."""

    __slots__ = ("s_tot", "l_tot", "j_tot")
    quantum_number_names: ClassVar = ("i_c", "s_c", "l_c", "s_r", "l_r", "s_tot", "l_tot", "j_tot", "f_tot")
    coupled_quantum_numbers: ClassVar = {
        "s_tot": ("s_c", "s_r"),
        "l_tot": ("l_c", "l_r"),
        "j_tot": ("s_tot", "l_tot"),
        "f_tot": ("i_c", "j_tot"),
    }
    coupling_scheme = "LS"

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
        species: str | SpeciesObject | None = None,
    ) -> None:
        """Initialize the Spin ket."""
        super().__init__(i_c, s_c, l_c, s_r, l_r, f_tot, m, species)

        self.s_tot = try_trivial_spin_addition(self.s_c, self.s_r, s_tot, "s_tot")
        self.l_tot = int(try_trivial_spin_addition(self.l_c, self.l_r, l_tot, "l_tot"))
        self.j_tot = try_trivial_spin_addition(self.l_tot, self.s_tot, j_tot, "j_tot")
        self.f_tot = try_trivial_spin_addition(self.j_tot, self.i_c, f_tot, "f_tot")

        super()._post_init()

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if not check_spin_addition_rule(self.l_r, self.l_c, self.l_tot):
            msgs.append(f"{self.l_r=}, {self.l_c=}, {self.l_tot=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.s_r, self.s_c, self.s_tot):
            msgs.append(f"{self.s_r=}, {self.s_c=}, {self.s_tot=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.l_tot, self.s_tot, self.j_tot):
            msgs.append(f"{self.l_tot=}, {self.s_tot=}, {self.j_tot=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.j_tot, self.i_c, self.f_tot):
            msgs.append(f"{self.j_tot=}, {self.i_c=}, {self.f_tot=} don't satisfy spin addition rule.")

        super().sanity_check(msgs)


class AngularKetJJ(AngularKetBase):
    """Spin ket in JJ coupling."""

    __slots__ = ("j_c", "j_r", "j_tot")
    quantum_number_names: ClassVar = ("i_c", "s_c", "l_c", "s_r", "l_r", "j_c", "j_r", "j_tot", "f_tot")
    coupled_quantum_numbers: ClassVar = {
        "j_c": ("s_c", "l_c"),
        "j_r": ("s_r", "l_r"),
        "j_tot": ("j_c", "j_r"),
        "f_tot": ("i_c", "j_tot"),
    }
    coupling_scheme = "JJ"

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
        species: str | SpeciesObject | None = None,
    ) -> None:
        """Initialize the Spin ket."""
        super().__init__(i_c, s_c, l_c, s_r, l_r, f_tot, m, species)

        self.j_c = try_trivial_spin_addition(self.l_c, self.s_c, j_c, "j_c")
        self.j_r = try_trivial_spin_addition(self.l_r, self.s_r, j_r, "j_r")
        self.j_tot = try_trivial_spin_addition(self.j_c, self.j_r, j_tot, "j_tot")
        self.f_tot = try_trivial_spin_addition(self.j_tot, self.i_c, f_tot, "f_tot")

        super()._post_init()

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if not check_spin_addition_rule(self.l_c, self.s_c, self.j_c):
            msgs.append(f"{self.l_c=}, {self.s_c=}, {self.j_c=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.l_r, self.s_r, self.j_r):
            msgs.append(f"{self.l_r=}, {self.s_r=}, {self.j_r=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.j_c, self.j_r, self.j_tot):
            msgs.append(f"{self.j_c=}, {self.j_r=}, {self.j_tot=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.j_tot, self.i_c, self.f_tot):
            msgs.append(f"{self.j_tot=}, {self.i_c=}, {self.f_tot=} don't satisfy spin addition rule.")

        super().sanity_check(msgs)


class AngularKetFJ(AngularKetBase):
    """Spin ket in FJ coupling."""

    __slots__ = ("j_c", "f_c", "j_r")
    quantum_number_names: ClassVar = ("i_c", "s_c", "l_c", "s_r", "l_r", "j_c", "f_c", "j_r", "f_tot")
    coupled_quantum_numbers: ClassVar = {
        "j_c": ("s_c", "l_c"),
        "f_c": ("i_c", "j_c"),
        "j_r": ("s_r", "l_r"),
        "f_tot": ("f_c", "j_r"),
    }
    coupling_scheme = "FJ"

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
        species: str | SpeciesObject | None = None,
    ) -> None:
        """Initialize the Spin ket."""
        super().__init__(i_c, s_c, l_c, s_r, l_r, f_tot, m, species)

        self.j_c = try_trivial_spin_addition(self.l_c, self.s_c, j_c, "j_c")
        self.j_r = try_trivial_spin_addition(self.l_r, self.s_r, j_r, "j_r")
        self.f_c = try_trivial_spin_addition(self.j_c, self.i_c, f_c, "f_c")
        self.f_tot = try_trivial_spin_addition(self.f_c, self.j_r, f_tot, "f_tot")

        super()._post_init()

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if not check_spin_addition_rule(self.l_c, self.s_c, self.j_c):
            msgs.append(f"{self.l_c=}, {self.s_c=}, {self.j_c=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.l_r, self.s_r, self.j_r):
            msgs.append(f"{self.l_r=}, {self.s_r=}, {self.j_r=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.j_c, self.i_c, self.f_c):
            msgs.append(f"{self.j_c=}, {self.i_c=}, {self.f_c=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.f_c, self.j_r, self.f_tot):
            msgs.append(f"{self.f_c=}, {self.j_r=}, {self.f_tot=} don't satisfy spin addition rule.")

        super().sanity_check(msgs)
