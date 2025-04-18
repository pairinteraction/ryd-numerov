from ryd_numerov.elements.element import Element


class Rubidium(Element):
    species = "Rb"
    s = 1 / 2
    ground_state_shell = (5, 0)

    ionization_energy_ghz = 1_010_029.164_6  # ±0.000_3
    # https://journals.aps.org/pra/pdf/10.1103/PhysRevA.83.052515
    # and see also https://webbook.nist.gov/cgi/inchi?ID=C7440177&Mask=20
