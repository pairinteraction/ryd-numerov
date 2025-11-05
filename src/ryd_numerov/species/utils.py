import math
import re


def calc_nu_from_energy(reduced_mass_factor: float, energy_au: float) -> float:
    r"""Calculate the effective principal quantum number nu from a given energy.

    The effective principal quantum number is given by

    .. math::
        \nu = \sqrt{\frac{1}{2} \frac{\mu}{-E}}

    where :math:`\mu` is the reduced mass factor and :math:`E` the energy in atomic units.

    Args:
        reduced_mass_factor: The reduced mass factor :math:`\mu = \frac{m_{Core}}{m_{Core} + m_e}`.
        energy_au: The energy :math:`E` in atomic units (hartree).

    Returns:
        The effective principal quantum number :math:`\nu`.

    """
    nu = math.sqrt(0.5 * reduced_mass_factor / -energy_au)
    if abs(nu - round(nu)) < 1e-10:
        nu = round(nu)
    return nu


def calc_energy_from_nu(reduced_mass_factor: float, nu: float) -> float:
    r"""Calculate the energy from a given effective principal quantum number nu.

    The energy is given by

    .. math::
        E = -\frac{1}{2} \frac{\mu}{\nu^2}

    where :math:`\mu` is the reduced mass factor and :math:`\nu` the effective principal quantum number.

    Args:
        reduced_mass_factor: The reduced mass factor :math:`\mu = \frac{m_{Core}}{m_{Core} + m_e}`.
        nu: The effective principal quantum number :math:`\nu`.

    Returns:
        The energy :math:`E` in atomic units (hartree).

    """
    return -0.5 * reduced_mass_factor / nu**2


def convert_electron_configuration(config: str) -> list[tuple[int, int, int]]:
    """Convert an electron configuration string to a list of tuples [(n, l, number), ...].

    This means convert a string representing the outermost electrons
    like "4f14.6s" to [(4, 2, 14), (6, 0, 1)].
    """
    l_str2int = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "k": 7, "l": 8, "m": 9}
    parts = config.split(".")
    converted_parts = []
    for part in parts:
        match = re.match(r"^(\d+)([a-z])(\d*)$", part)
        if match is None:
            raise ValueError(f"Invalid configuration format: {config}.")
        n = int(match.group(1))
        l = l_str2int[match.group(2)]
        number = int(match.group(3)) if match.group(3) else 1
        converted_parts.append((n, l, number))

    return converted_parts
