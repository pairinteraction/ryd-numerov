from ryd_numerov.elements.element import Element


class Potassium(Element):
    species = "K"
    s = 1 / 2
    ground_state_shell = (4, 0)

    _ionization_energy = (4.340_66, 0.000_01, "eV")
    # https://webbook.nist.gov/cgi/inchi?ID=C7440097&Mask=20
