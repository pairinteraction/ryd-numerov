from ryd_numerov.elements.element import Element


class Hydrogen(Element):
    species = "H"
    s = 1 / 2
    ground_state_shell = (1, 0)

    _ionization_energy = (15.425_93, 0.000_05, "eV")
    # https://webbook.nist.gov/cgi/inchi?ID=C1333740&Mask=20
