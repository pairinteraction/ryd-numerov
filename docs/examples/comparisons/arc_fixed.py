import arc
import numpy as np
import scipy.integrate


def radialWavefunction(
    atom: arc.AlkaliAtom, n: int, l: int, j: float, step: float = 1e-2, use_fixed_arc: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    n, l, j = int(n), int(l), float(j)
    E_H = 27.211 if not use_fixed_arc else 27.211_386_245_988
    energy = atom.getEnergy(int(n), int(l), float(j)) / E_H
    r_list, psi_r_list = atom.radialWavefunction(
        int(l),
        0.5,
        j,
        energy,
        atom.alphaC ** (1 / 3.0),
        2.0 * n * (n + 15.0),
        step=step,
    )

    # Fixup the sign of the wavefunction, such that the first amplitude is positive
    r, psi_r = np.array(r_list), np.array(psi_r_list)
    if psi_r[np.argwhere(psi_r != 0)[0][0]] < 0:
        psi_r = -psi_r

    return r, psi_r


def getRadialMatrixElement(
    atom: arc.AlkaliAtom,
    n1: int,
    l1: int,
    j1: float,
    n2: int,
    l2: int,
    j2: float,
    step: float = 1e-2,
    use_fixed_arc: bool = False,
) -> float:
    r1, psi1_r1 = radialWavefunction(atom, n1, l1, j1, step, use_fixed_arc)
    r2, psi2_r2 = radialWavefunction(atom, n2, l2, j2, step, use_fixed_arc)

    upTo = min(len(r1), len(r2))
    return scipy.integrate.trapezoid(
        np.multiply(np.multiply(psi1_r1[0:upTo], psi2_r2[0:upTo]), r1[0:upTo]),
        x=r1[0:upTo],
    )
