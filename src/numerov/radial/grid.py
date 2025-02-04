from typing import Optional

import numpy as np


class Grid:
    """A grid object storing all relevant information about the grid points.

    We store the grid in the dimensionless form x = r/a_0, as well as in the scaled dimensionless form z = sqrt{x}.
    The benefit of using z is that the nodes of the wavefunction are equally spaced in z-space,
    allowing for a computational better choice of choosing the constant step size during the integration.
    """

    def __init__(
        self,
        zmin: float,
        zmax: float,
        dz: float,
    ) -> None:
        """Initialize the grid object.

        Args:
            zmin: The minimum value of the scaled dimensionless coordinate z = sqrt{x}.
            zmax: The maximum value of the scaled dimensionless coordinate z = sqrt{x}.
            dz: The step size of the grid in the scaled dimensionless coordinate z = sqrt{x}
            (exactly one of dz or steps must be provided).
            steps: The number of steps in the grid (exactly one of dz or steps must be provided).

        """
        self._dz = dz
        self._zlist = np.arange(zmin, zmax + dz * 1e-3, dz)

    def __len__(self) -> int:
        return self.steps

    def __repr__(self) -> str:
        return f"Grid({self.zmin}, {self.zmax}, dz={self.dz}, steps={self.steps})"

    @property
    def steps(self) -> int:
        """The number of steps in the grid."""
        return len(self.zlist)

    @property
    def dz(self) -> float:
        """The step size of the grid in the scaled dimensionless coordinate z = sqrt{x}."""
        return self._dz

    @property
    def zmin(self) -> float:
        """The minimum value of the scaled dimensionless coordinate z = sqrt{x}."""
        return self.zlist[0]

    @property
    def zmax(self) -> float:
        """The maximum value of the scaled dimensionless coordinate z = sqrt{x}."""
        return self.zlist[-1]

    @property
    def zlist(self) -> np.ndarray:
        """The grid in the scaled dimensionless coordinate z = sqrt{x}."""
        return self._zlist

    @property
    def xmin(self) -> float:
        """The minimum value of the dimensionless coordinate x = r/a_0."""
        return self.zmin**2

    @property
    def xmax(self) -> float:
        """The maximum value of the dimensionless coordinate x = r/a_0."""
        return self.zmax**2

    @property
    def xlist(self) -> np.ndarray:
        """The grid in the dimensionless coordinate x = r/a_0."""
        return self.zlist**2

    def set_grid_range(self, step_start: Optional[int] = None, step_stop: Optional[int] = None) -> None:
        """Restrict the grid to the range [step_start, step_stop]."""
        if step_start is None:
            step_start = 0
        if step_stop is None:
            step_stop = self.steps
        self._zlist = self._zlist[step_start:step_stop]
