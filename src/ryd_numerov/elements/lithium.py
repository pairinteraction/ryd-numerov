from ryd_numerov.elements.element import Element


class Lithium(Element):
    species = "Li"
    s = 1 / 2
    ground_state_shell = (2, 0)

    # https://webbook.nist.gov/cgi/inchi?ID=C7439932&Mask=20
    _ionization_energy = (5.391_72, None, "eV")
