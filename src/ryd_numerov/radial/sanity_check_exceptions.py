"""The numerov integration only works well if the energies and model potentials are a good approximation.

For different elements (especially for earth alkali metals) some low lying states do not converge well.
In wavefunction.py we added sanity checks, that the wavefunctions behaves as expected.

In this module we define states, where we know that the numerov integration does not converge perfectly,
but we still want to use them and ignore the warnings.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ryd_numerov.model import ModelPotential


def is_known_exception(model_potential: "ModelPotential", warning_msgs: list[str]) -> bool:
    """Check if the state is a known exception."""
    species = model_potential.species
    n = model_potential.n
    l = model_potential.l
    j = model_potential.j

    if (
        species in ["Li", "Na", "K", "Rb", "Cs"]
        and n <= 11
        and len(warning_msgs) == 1
        and "inner_weight_scaled_to_whole_grid" in warning_msgs[0]
    ):
        return True

    if species in ["Sr_singlet", "Sr_triplet"] and 11 <= n <= 1000:
        if l in [4, 5] and len(warning_msgs) == 1 and "inner_weight_scaled_to_whole_grid" in warning_msgs[0]:
            return True

        if (
            l in [1, 2, 3]
            # and len(warning_msgs) == 3
            and "Wavefunction diverges at the inner boundary" in warning_msgs[0]
            and "Trying to correct the wavefunction." in warning_msgs[1]
            # and "The wavefunction is not close to zero at the inner boundary" in warning_msgs[2]
        ):
            return True

        if l in [0, 2] and "The maximum of the wavefunction is close to the inner boundary" in warning_msgs[0]:
            return True

    know_exceptions = [  # (species, n, l, j)
        ("Sr_singlet", 11, 2, 2),
        ("Sr_singlet", 12, 2, 2),
        ("Sr_triplet", 16, 2, 3),
    ]

    return (species, n, l, j) in know_exceptions
