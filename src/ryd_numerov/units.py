from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Union

from pint import UnitRegistry

if TYPE_CHECKING:
    import numpy.typing as npt
    from pint.facets.plain import PlainQuantity, PlainUnit
    from typing_extensions import TypeAlias

    NDArray: TypeAlias = npt.NDArray[Any]
    PintFloat: TypeAlias = PlainQuantity[float]
    PintArray: TypeAlias = PlainQuantity[NDArray]
    PintComplex: TypeAlias = PlainQuantity[complex]

ureg: UnitRegistry[float] = UnitRegistry(system="atomic")


MatrixElementOperator = Literal[
    "magnetic_dipole",
    "electric_dipole",
    "electric_quadrupole",
    "electric_octupole",
    "electric_quadrupole_zero",
]
MatrixElementOperatorRanks: dict[MatrixElementOperator, tuple[int, int]] = {
    # "operator": (k_radial, k_angular)
    "magnetic_dipole": (0, 1),
    "electric_dipole": (1, 1),
    "electric_quadrupole": (2, 2),
    "electric_octupole": (3, 3),
    "electric_quadrupole_zero": (2, 0),
}

Dimension = Literal[
    MatrixElementOperator,
    "electric_field",
    "magnetic_field",
    "distance",
    "energy",
    "charge",
    "velocity",
    "temperature",
    "time",
    "radial_matrix_element",
    "angular_matrix_element",
    "arbitrary",
    "zero",
]
DimensionLike = Union[Dimension, tuple[Dimension, Dimension]]

# some abbreviations: au_time: atomic_unit_of_time; au_current: atomic_unit_of_current; m_e: electron_mass
_CommonUnits: dict[Dimension, str] = {
    "electric_field": "V/cm",  # 1 V/cm = 1.9446903811524456e-10 bohr * m_e / au_current / au_time ** 3
    "magnetic_field": "T",  # 1 T = 4.254382157342044e-06 m_e / au_current / au_time ** 2
    "distance": "micrometer",  # 1 mum = 18897.26124622279 bohr
    "energy": "hartree",  # 1 hartree = 1 bohr ** 2 * m_e / au_time ** 2
    "charge": "e",  # 1 e = 1 au_current * au_time
    "velocity": "speed_of_light",  # 1 c = 137.03599908356244 bohr / au_time
    "temperature": "K",  # 1 K = 3.1668115634555572e-06 atomic_unit_of_temperature
    "time": "s",  # 1 s = 4.134137333518244e+16 au_time
    "radial_matrix_element": "bohr",  # 1 bohr
    "angular_matrix_element": "",  # 1 dimensionless
    "electric_dipole": "e * a0",  # 1 e * a0 = 1 au_current * au_time * bohr
    "electric_quadrupole": "e * a0^2",  # 1 e * a0^2 = 1 au_current * au_time * bohr ** 2
    "electric_quadrupole_zero": "e * a0^2",  # 1 e * a0^2 = 1 au_current * au_time * bohr ** 2
    "electric_octupole": "e * a0^3",  # 1 e * a0^3 = 1 au_current * au_time * bohr ** 3
    "magnetic_dipole": "bohr_magneton",  # 1 bohr_magneton = 0.5 au_current * bohr ** 2'
    "arbitrary": "",  # 1 dimensionless
    "zero": "",  # 1 dimensionless
}
BaseUnits: dict[Dimension, PlainUnit] = {
    k: ureg.Quantity(1, unit).to_base_units().units for k, unit in _CommonUnits.items()
}
BaseQuantities: dict[Dimension, PintFloat] = {k: ureg.Quantity(1, unit) for k, unit in BaseUnits.items()}

Context = Literal["spectroscopy", "Gaussian"]
BaseContexts: dict[Dimension, Context] = {
    "magnetic_field": "Gaussian",
    "energy": "spectroscopy",
}


rydberg_constant = ureg.Quantity(1, "rydberg_constant").to("hartree", "spectroscopy")
electron_mass = ureg.Quantity(1, "electron_mass").to("u")
