from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ryd_numerov.units import NDArray


class Grid:
    """A grid object storing all relevant information about the grid points.

    We store the grid in the dimensionless form x = r/a_0, as well as in the scaled dimensionless form z = sqrt{x}.
    The benefit of using z is that the nodes of the wavefunction are equally spaced in z-space,
    allowing for a computational better choice of choosing the constant step size during the integration.
    """

    def __init__(
        self,
        z_min: float,
        z_max: float,
        dz: float,
    ) -> None:
        """Initialize the grid object.

        Args:
            z_min: The minimum value of the scaled dimensionless coordinate z = sqrt{x}.
            z_max: The maximum value of the scaled dimensionless coordinate z = sqrt{x}.
            dz: The step size of the grid in the scaled dimensionless coordinate z = sqrt{x}
            (exactly one of dz or steps must be provided).
            steps: The number of steps in the grid (exactly one of dz or steps must be provided).

        """
        self._dz = dz
        # put all grid points on a standard grid, i.e. [dz, 2*dz, 3*dz, ...]
        # this is necessary to allow integration of two different wavefunctions
        # Note: using np.arange((z_min // dz) * dz, z_max + dz / 2, dz)
        # would lead to 'quite big' inprecisions (1e-10) between grid points of different grids,
        # because of floating point errors
        self._z_list: NDArray = np.arange(0, z_max + dz / 2, dz)[round(z_min / dz) :]

    def __len__(self) -> int:
        return self.steps

    def __repr__(self) -> str:
        return f"Grid({self.z_min}, {self.z_max}, dz={self.dz}, steps={self.steps})"

    @property
    def steps(self) -> int:
        """The number of steps in the grid."""
        return len(self.z_list)

    @property
    def dz(self) -> float:
        """The step size of the grid in the scaled dimensionless coordinate z = sqrt{x}."""
        return self._dz

    @property
    def z_min(self) -> float:
        """The minimum value of the scaled dimensionless coordinate z = sqrt{x}."""
        return self.z_list[0]  # type: ignore [no-any-return]  # FIXME: numpy indexing

    @property
    def z_max(self) -> float:
        """The maximum value of the scaled dimensionless coordinate z = sqrt{x}."""
        return self.z_list[-1]  # type: ignore [no-any-return]  # FIXME: numpy indexing

    @property
    def z_list(self) -> NDArray:
        """The grid in the scaled dimensionless coordinate z = sqrt{x}.

        In this coordinate the grid points are chosen equidistant,
        because the nodes of the wavefunction are equally spaced in this coordinate.
        """
        return self._z_list

    @property
    def x_min(self) -> float:
        """The minimum value of the dimensionless coordinate x = r/a_0."""
        return self.z_min**2

    @property
    def x_max(self) -> float:
        """The maximum value of the dimensionless coordinate x = r/a_0."""
        return self.z_max**2

    @property
    def x_list(self) -> NDArray:
        """The grid in the dimensionless coordinate x = r/a_0."""
        return self.z_list**2

    def set_grid_range(self, step_start: int | None = None, step_stop: int | None = None) -> None:
        """Restrict the grid to the range [step_start, step_stop]."""
        if step_start is None:
            step_start = 0
        if step_stop is None:
            step_stop = self.steps
        self._z_list = self._z_list[step_start:step_stop]
