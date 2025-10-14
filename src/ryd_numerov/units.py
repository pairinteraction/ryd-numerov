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
    # type ignore because pint has no type support for complex
    PintComplex: TypeAlias = PlainQuantity[complex]  # type: ignore [type-var]

ureg = UnitRegistry(system="atomic")


OperatorType = Literal["MAGNETIC", "ELECTRIC", "SPHERICAL", "MAGNETIC_S", "MAGNETIC_L"]
MatrixElementType = Literal[
    "MAGNETIC_DIPOLE",  # MAGNETIC with k_radial = 0, k_angular = 1
    "ELECTRIC_DIPOLE",  # ELECTRIC with k_radial = 1, k_angular = 1
    "ELECTRIC_QUADRUPOLE",  # ELECTRIC with k_radial = 2, k_angular = 2
    "ELECTRIC_OCTUPOLE",  # ELECTRIC with k_radial = 3, k_angular = 3
    "ELECTRIC_QUADRUPOLE_ZERO",  # ELECTRIC with k_radial = 2, k_angular = 0
]

Dimension = Literal[
    "ELECTRIC_FIELD",
    "MAGNETIC_FIELD",
    "DISTANCE",
    "ENERGY",
    "CHARGE",
    "VELOCITY",
    "TEMPERATURE",
    "TIME",
    "RADIAL_MATRIX_ELEMENT",
    "ANGULAR_MATRIX_ELEMENT",
    "ELECTRIC_DIPOLE",
    "ELECTRIC_QUADRUPOLE",
    "ELECTRIC_QUADRUPOLE_ZERO",
    "ELECTRIC_OCTUPOLE",
    "MAGNETIC_DIPOLE",
    "ARBITRARY",
    "ZERO",
]
DimensionLike = Union[Dimension, tuple[Dimension, Dimension]]

# some abbreviations: au_time: atomic_unit_of_time; au_current: atomic_unit_of_current; m_e: electron_mass
_CommonUnits: dict[Dimension, str] = {
    "ELECTRIC_FIELD": "V/cm",  # 1 V/cm = 1.9446903811524456e-10 bohr * m_e / au_current / au_time ** 3
    "MAGNETIC_FIELD": "T",  # 1 T = 4.254382157342044e-06 m_e / au_current / au_time ** 2
    "DISTANCE": "micrometer",  # 1 mum = 18897.26124622279 bohr
    "ENERGY": "hartree",  # 1 hartree = 1 bohr ** 2 * m_e / au_time ** 2
    "CHARGE": "e",  # 1 e = 1 au_current * au_time
    "VELOCITY": "speed_of_light",  # 1 c = 137.03599908356244 bohr / au_time
    "TEMPERATURE": "K",  # 1 K = 3.1668115634555572e-06 atomic_unit_of_temperature
    "TIME": "s",  # 1 s = 4.134137333518244e+16 au_time
    "RADIAL_MATRIX_ELEMENT": "bohr",  # 1 bohr
    "ANGULAR_MATRIX_ELEMENT": "",  # 1 dimensionless
    "ELECTRIC_DIPOLE": "e * a0",  # 1 e * a0 = 1 au_current * au_time * bohr
    "ELECTRIC_QUADRUPOLE": "e * a0^2",  # 1 e * a0^2 = 1 au_current * au_time * bohr ** 2
    "ELECTRIC_QUADRUPOLE_ZERO": "e * a0^2",  # 1 e * a0^2 = 1 au_current * au_time * bohr ** 2
    "ELECTRIC_OCTUPOLE": "e * a0^3",  # 1 e * a0^3 = 1 au_current * au_time * bohr ** 3
    "MAGNETIC_DIPOLE": "bohr_magneton",  # 1 bohr_magneton = 0.5 au_current * bohr ** 2'
    "ARBITRARY": "",  # 1 dimensionless
    "ZERO": "",  # 1 dimensionless
}
BaseUnits: dict[Dimension, PlainUnit] = {
    k: ureg.Quantity(1, unit).to_base_units().units for k, unit in _CommonUnits.items()
}
BaseQuantities: dict[Dimension, PintFloat] = {k: ureg.Quantity(1, unit) for k, unit in BaseUnits.items()}

Context = Literal["spectroscopy", "Gaussian"]
BaseContexts: dict[Dimension, Context] = {
    "MAGNETIC_FIELD": "Gaussian",
    "ENERGY": "spectroscopy",
}


rydberg_constant = ureg.Quantity(1, "rydberg_constant").to("hartree", "spectroscopy")
electron_mass = ureg.Quantity(1, "electron_mass").to("u")
