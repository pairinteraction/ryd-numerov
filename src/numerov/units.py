from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pint import UnitRegistry

if TYPE_CHECKING:
    from pint.facets.plain import PlainUnit

    Array = np.ndarray[Any, Any]

ureg = UnitRegistry(system="atomic")


Dimension = Literal[
    "ELECTRIC_FIELD",
    "MAGNETIC_FIELD",
    "DISTANCE",
    "ENERGY",
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
BaseUnits: dict[Dimension, "PlainUnit"] = {
    # ELECTRIC_FIELD: 1 V/cm = 1.94469038e-10 electron_mass * bohr / atomic_unit_of_time ** 3 / atomic_unit_of_current
    "ELECTRIC_FIELD": ureg.Quantity(1, "V/cm").to_base_units().units,
    # MAGNETIC_FIELD: 1 T = 4.25438216e-06 electron_mass / atomic_unit_of_time ** 2 / atomic_unit_of_current
    "MAGNETIC_FIELD": ureg.Quantity(1, "T").to_base_units().units,
    # DISTANCE: 1 mum = 18897.2612 bohr
    "DISTANCE": ureg.Quantity(1, "micrometer").to_base_units().units,
    # ENERGY: 1 hartree = 1 electron_mass * bohr ** 2 / atomic_unit_of_time ** 2
    "ENERGY": ureg.Unit("hartree"),
    # DISTANCE: 1 mum = 18897.2612 bohr
    "RADIAL_MATRIX_ELEMENT": ureg.Unit("bohr"),
    #
    "ANGULAR_MATRIX_ELEMENT": ureg.Unit(""),
    # ELECTRIC_DIPOLE: 1 e * a0 = 1 atomic_unit_of_current * atomic_unit_of_time * bohr
    "ELECTRIC_DIPOLE": ureg.Quantity(1, "e * a0").to_base_units().units,
    # ELECTRIC_QUADRUPOLE: 1 e * a0^2 = 1 atomic_unit_of_current * atomic_unit_of_time * bohr ** 2
    "ELECTRIC_QUADRUPOLE": ureg.Quantity(1, "e * a0^2").to_base_units().units,
    # ELECTRIC_QUADRUPOLE_ZERO: 1 e * a0^2 = 1 atomic_unit_of_current * atomic_unit_of_time * bohr ** 2
    "ELECTRIC_QUADRUPOLE_ZERO": ureg.Quantity(1, "e * a0^2").to_base_units().units,
    # ELECTRIC_OCTUPOLE: 1 e * a0^3 = 1 atomic_unit_of_current * atomic_unit_of_time * bohr ** 3
    "ELECTRIC_OCTUPOLE": ureg.Quantity(1, "e * a0^3").to_base_units().units,
    # MAGNETIC_DIPOLE: 1 hbar e / m_e = 1 bohr ** 2 * atomic_unit_of_current
    "MAGNETIC_DIPOLE": ureg.Quantity(1, "hbar e / m_e").to_base_units().units,
    "ARBITRARY": ureg.Unit(""),
    "ZERO": ureg.Unit(""),
}
