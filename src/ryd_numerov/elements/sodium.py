from ryd_numerov.elements.element import Element


class Sodium(Element):
    species = "Na"
    s = 1 / 2
    ground_state_shell = (3, 0)

    _ionization_energy = (5.139_08, None, "eV")
    # https://webbook.nist.gov/cgi/inchi?ID=C7440235&Mask=20
