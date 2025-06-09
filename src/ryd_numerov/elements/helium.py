from ryd_numerov.elements.element import Element


class HeliumIon(Element):
    species = "He+"
    Z = 2
    s = 1 / 2
    ground_state_shell = (1, 0)

    # https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=He&units=1&at_num_out=on&el_name_out=on&seq_out=on&shells_out=on&level_out=on&e_out=0&unc_out=on&biblio=on
    _ionization_energy = (54.417_765_528_2, 0.000_000_001_0, "eV")

    _corrected_rydberg_constant = (109722.38680085634, None, "1/cm")
