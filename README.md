# Rydberg Numerov

[![License: LGPL v3][license-lgpl-svg]][license-lgpl-link]
[![CI Workflow][gh-workflow-svg]][gh-workflow-link]
[![Documentation][docs-svg]][docs-link]

[license-lgpl-svg]: https://img.shields.io/badge/License-LGPL_v3-blue.svg?style=flat
[license-lgpl-link]: https://www.gnu.org/licenses/lgpl-3.0.html
[gh-workflow-svg]: https://github.com/pairinteraction/ryd-numerov/actions/workflows/python_wheel.yml/badge.svg
[gh-workflow-link]: https://github.com/pairinteraction/ryd-numerov/actions/workflows/python_wheel.yml
[docs-svg]: https://img.shields.io/badge/Documentation-rydnumerov-blue.svg?style=flat
[docs-link]: https://www.pairinteraction.org/ryd-numerov/sphinx/html/

The *Rydberg Numerov* software calculates properties of Rydberg states.
We especially focus on the calculation of the radial wavefunction of Rydberg states via the Numerov method.
The software can be installed via pip (requires Python >= 3.9):

```bash
pip install pairinteraction
```

To install the latest development version from github, use:

```bash
pip install git+https://github.com/pairinteraction/ryd-numerov
```


## How to Cite

This package relies on quantum defects provided by the community. Consider citing relevant publications for your atomic species.

<p><details>
<summary><b>Click to expand for quantum defect references</b></summary>

| Element | Model                 | Identifier     | References                                                                                                                                                   |
|---------|-----------------------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| H       | SQDT                  | `H`            | Schrödinger equation for hydrogen                                                                                                                            |
| Li      | SQDT                  | `Li`           | [10.1017/CBO9780511524530] (1994)<br>[10.1103/PhysRevA.34.2889] (1986)                                                                                       |
| Na      | SQDT                  | `Na`           | [10.1088/0953-4075/30/10/009] (1997)<br>[10.1070/QE1995v025n09ABEH000501] (1995)<br>[10.1103/PhysRevA.45.4720] (1992)                                        |
| K       | SQDT                  | `K`            | [10.1088/0031-8949/27/4/012] (1983)<br>[10.1016/0030-4018(81)90225-X] (1981)                                                                                 |
| Rb      | SQDT                  | `Rb`           | [10.1103/PhysRevA.83.052515] (2011)<br>[10.1103/PhysRevA.74.054502] (2006)<br>[10.1103/PhysRevA.74.062712] (2006)<br>[10.1103/PhysRevA.67.052502] (2003)     |
| Cs      | SQDT                  | `Cs`           | [10.1103/PhysRevA.93.013424] (2016)<br>[10.1103/PhysRevA.35.4650] (1987)<br>[10.1103/PhysRevA.26.2733] (1982)                                                |
| Sr88    | SQDT, singlet sector  | `Sr88_singlet` | [10.1103/PhysRevA.108.022815] (2023)<br>[10.17169/refubium-34581] (2022)                                                                                     |
| Sr88    | SQDT, triplet sector  | `Sr88_triplet` | [10.1016/j.cpc.2020.107814] (2021)                                                                                                                           |
</details></p>

[10.1103/PhysRevA.34.2889]: https://doi.org/10.1103/PhysRevA.34.2889
[10.1017/CBO9780511524530]: https://doi.org/10.1017/CBO9780511524530
[10.1103/PhysRevA.45.4720]: https://doi.org/10.1103/PhysRevA.45.4720
[10.1070/QE1995v025n09ABEH000501]: https://doi.org/10.1070/QE1995v025n09ABEH000501
[10.1088/0953-4075/30/10/009]: https://doi.org/10.1088/0953-4075/30/10/009
[10.1088/0031-8949/27/4/012]: https://doi.org/10.1088/0031-8949/27/4/012
[10.1016/0030-4018(81)90225-X]: https://doi.org/10.1016/0030-4018(81)90225-X
[10.1103/PhysRevA.83.052515]: https://doi.org/10.1103/PhysRevA.83.052515
[10.1103/PhysRevA.67.052502]: https://doi.org/10.1103/PhysRevA.67.052502
[10.1103/PhysRevA.74.054502]: https://doi.org/10.1103/PhysRevA.74.054502
[10.1103/PhysRevA.74.062712]: https://doi.org/10.1103/PhysRevA.74.062712
[10.1103/PhysRevA.93.013424]: https://doi.org/10.1103/PhysRevA.93.013424
[10.1103/PhysRevA.26.2733]: https://doi.org/10.1103/PhysRevA.26.2733
[10.1103/PhysRevA.35.4650]: https://doi.org/10.1103/PhysRevA.35.4650
[10.1103/PhysRevA.108.022815]: https://doi.org/10.1103/PhysRevA.108.022815
[10.17169/refubium-34581]: https://doi.org/10.17169/refubium-34581
[10.1016/j.cpc.2020.107814]: https://doi.org/10.1016/j.cpc.2020.107814


## Documentation

**User Guide**

- [Tutorials] - Examples of how to use the Rydberg Numerov library.

- [API Reference] - Documentation of classes and functions of the Rydberg Numerov Python library.


[Tutorials]: https://www.pairinteraction.org/ryd-numerov/sphinx/html/examples.html
[API Reference]: https://www.pairinteraction.org/ryd-numerov/sphinx/html/modules.html


## License

The ryd-numerov software is licensed under [LGPL v3][license-lgpl-link]. For more information, see [LICENSE.txt](https://github.com/pairinteraction/ryd-numerov/blob/master/LICENSE.txt).
