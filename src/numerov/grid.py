from functools import cached_property
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
        dz: Optional[float],
        steps: Optional[int],
    ) -> None:
        """Initialize the grid object.

        Args:
            zmin: The minimum value of the scaled dimensionless coordinate z = sqrt{x}.
            zmax: The maximum value of the scaled dimensionless coordinate z = sqrt{x}.
            dz: The step size of the grid in the scaled dimensionless coordinate z = sqrt{x}
            (exactly one of dz or steps must be provided).
            steps: The number of steps in the grid (exactly one of dz or steps must be provided).

        """
        if dz is None and steps is None:
            raise ValueError("Either dz or steps must be provided.")

        if dz is None:
            dz = (zmax - zmin) / (steps - 1)
        self.zmin = zmin
        self.zmax = zmax
        self.dz = dz
        self.zlist = np.arange(zmin, zmax + dz, dz)
        self.steps = len(self.zlist)

        if steps is not None and self.steps != steps:
            raise ValueError("Only one of dz or steps should be provided (or they should match).")

    def __len__(self) -> int:
        return self.steps

    def __repr__(self) -> str:
        return f"Grid({self.zmin}, {self.zmax}, dz={self.dz}, steps={self.steps})"

    @cached_property
    def xmin(self) -> float:
        """The minimum value of the dimensionless coordinate x = r/a_0."""
        return self.zmin**2

    @cached_property
    def xmax(self) -> float:
        """The maximum value of the dimensionless coordinate x = r/a_0."""
        return self.zmax**2

    @cached_property
    def xlist(self) -> np.ndarray:
        """The grid in the dimensionless coordinate x = r/a_0."""
        return self.zlist**2
