import pytest

from ryd_numerov.elements import BaseElement
from ryd_numerov.rydberg import RydbergState


@pytest.mark.parametrize("species", BaseElement.get_available_species())
def test_magnetic(species: str) -> None:
    """Test magnetic units."""
    element = BaseElement.from_species(species)

    if element.number_valence_electrons == 1:
        ket = RydbergState(species, n=50, l=0)
        ket.create_wavefunction()
        with pytest.raises(AssertionError, match="j_tot must be set"):
            RydbergState(species, n=50, l=1)

    elif element.number_valence_electrons == 2 and element._quantum_defects is not None:  # noqa: SLF001
        for s_tot in [0, 1]:
            ket = RydbergState(species, n=50, l=1, j_tot=1 + s_tot, s_tot=s_tot)
            ket.create_wavefunction()
