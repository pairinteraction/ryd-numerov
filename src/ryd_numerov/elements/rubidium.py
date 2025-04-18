from ryd_numerov.elements.element import Element


class Rubidium(Element):
    species = "Rb"
    s = 1 / 2
    ground_state_shell = (5, 0)

    _ionization_energy = (1_010_029.164_6, 0.000_3, "GHz")
    # https://journals.aps.org/pra/pdf/10.1103/PhysRevA.83.052515

    # older value
    # _ionization_energy = (4.177_13, 0.000_002, "eV")  # noqa: ERA001
    # corresponds to (1_010_025.54, 0.48, "GHz")
    # https://webbook.nist.gov/cgi/inchi?ID=C7440177&Mask=20
