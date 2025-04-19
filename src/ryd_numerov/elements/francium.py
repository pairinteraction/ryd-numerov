from ryd_numerov.elements.element import Element


class Francium(Element):
    species = "Fr"
    s = 1 / 2
    ground_state_shell = (7, 0)

    _ionization_energy = (4.071_2, 0.000_04, "eV")
    # https://webbook.nist.gov/cgi/inchi?ID=C7440735&Mask=20
