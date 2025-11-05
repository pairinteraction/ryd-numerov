from typing import TYPE_CHECKING

import pytest
from ryd_numerov.rydberg_state import RydbergStateAlkali, RydbergStateAlkalineLS
from ryd_numerov.species import BaseElement

if TYPE_CHECKING:
    from ryd_numerov.rydberg_state import RydbergStateBase


@pytest.mark.parametrize("species", BaseElement.get_available_species())
def test_magnetic(species: str) -> None:
    """Test magnetic units."""
    element = BaseElement.from_species(species)

    state: RydbergStateBase
    if element.number_valence_electrons == 1:
        state = RydbergStateAlkali(species, n=50, l=0)
        state.radial.create_wavefunction()
        with pytest.raises(ValueError, match="j must be set"):
            RydbergStateAlkali(species, n=50, l=1)
    elif element.number_valence_electrons == 2 and element._quantum_defects is not None:  # noqa: SLF001
        for s_tot in [0, 1]:
            state = RydbergStateAlkalineLS(species, n=50, l=1, s_tot=s_tot, j_tot=1 + s_tot)
            state.radial.create_wavefunction()
