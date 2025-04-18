from ryd_numerov.elements.element import Element


class Cesium(Element):
    species = "Cs"
    s = 1 / 2
    ground_state_shell = (6, 0)

    _ionization_energy = (3.893_90, 0.000_002, "eV")
    # https://webbook.nist.gov/cgi/inchi?ID=C7440462&Mask=20
