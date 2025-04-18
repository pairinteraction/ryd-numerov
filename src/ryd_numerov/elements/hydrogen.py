from ryd_numerov.elements.element import Element


class Hydrogen(Element):
    species = "H"
    s = 1 / 2
    ground_state_shell = (1, 0)
