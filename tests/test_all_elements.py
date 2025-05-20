import pytest

from ryd_numerov.rydberg import RydbergState


@pytest.mark.parametrize("species", ["H_textbook", "H", "Li", "Na", "K", "Rb", "Cs", "Sr88_singlet", "Sr88_triplet"])
def test_magnetic(species: str) -> None:
    """Test magnetic units."""
    ket = RydbergState(species, n=50, l=0)
    ket.create_wavefunction()

    if ket.s != 0:
        with pytest.raises(ValueError, match="j must be given"):
            RydbergState(species, n=50, l=1)

    ket2 = RydbergState(species, n=50, l=1, j=1 + ket.s)
    ket2.create_wavefunction()
