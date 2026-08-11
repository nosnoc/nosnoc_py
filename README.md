# `nosnoc`

[![Tests](https://img.shields.io/github/actions/workflow/status/nosnoc/nosnoc_py/tests.yml?label=tests)](https://github.com/nosnoc/nosnoc/actions)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://nosnoc-py.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/nosnoc/nosnoc_py)](https://github.com/nosnoc/nosnoc/blob/main/LICENSE)

## About
`nosnoc` is an open source Python software package for NOnSmooth Numerical Optimal Control.

A detailed overview of the theory and methods behind NOSNOC can be found in the course material of the
[Summer School on Direct Methods for Optimal Control of Nonsmooth Systems](https://www.syscop.de/teaching/ss2023/summer-school-direct-methods-optimal-control-nonsmooth-systems).

It is a structured reimplementation of the Matlab package NOSNOC (https://github.com/nurkanovic/nosnoc), but does not support all features in there (yet).
Most importantly, time freezing is not implemented yet.

It implements the FESD (Finite elements with switch detection) method, which allows one to handle nonsmooth ODE systems accurately in simulation and optimal control problems.

More information can be found in the NOSNOC package (https://github.com/nurkanovic/nosnoc).


## Installation
`nosnoc` is now available on `PyPI`! As such you can `pip` install it as you would any other package.

1. Setup virtual environment:
```
virtualenv env --python=python3
```

2. Source environment:
```
source env/bin/activate
```

3. Install
```
pip install nosnoc
```

### Optional additional installation steps
In case you are developing the `nosnoc` package you can install the following optional [dependency-groups](https://packaging.python.org/en/latest/specifications/dependency-groups/):

1. `docs`, for building the docs:
```
pip install --group docs -e .
```
2. In order to run tests you should install the testing requirements:
```
pip install --group test -e .
```

Note that for this you will need a sufficiently modern version of your python package manager (normally `pip`).
## Citing nosnoc

If you use **nosnoc** in research, please cite the software paper:

~~~bibtex
@article{Nurkanovic2022,
  title={nosnoc: A software package for numerical optimal control of nonsmooth systems},
  author={Nurkanovi{\'c}, Armin and Diehl, Moritz},
  journal={IEEE Control Systems Letters},
  volume={6},
  pages={3110--3115},
  year={2022},
  publisher={IEEE}
}
~~~

### Recommended additional citations

Depending on which features of **nosnoc** you use, please also cite the corresponding methodological papers.

#### Real-time MPC algorithms

~~~bibtex
@Article{Nurkanovic2026a,
  Title                    = {Real-Time Algorithms for Model Predictive Control of Hybrid Dynamical Systems},
  Author                   = {Nurkanovi{\'c}, Armin and Pozharskiy, Anton and Diehl, Moritz},
  Journal                  = {arXiv preprint},
  Year                     = {2026},
  Url                      = {https://www.syscop.de/files/users/armin.nurkanovic/Nurkanovic2026a.pdf}
}
~~~

#### FESD

~~~bibtex
@article{Nurkanovic2024,
  title={Finite elements with switch detection for direct optimal control of nonsmooth systems},
  author={Nurkanovi{\'c}, Armin and Sperl, Mario and Albrecht, Sebastian and Diehl, Moritz},
  journal={Numerische Mathematik},
  pages={1--48},
  year={2024},
  publisher={Springer}
}
~~~

~~~bibtex
@article{Nurkanovic2024a,
  title={Finite Elements with Switch Detection for numerical optimal control of nonsmooth dynamical systems with set-valued heaviside step functions},
  author={Nurkanovi{\'c}, Armin and Pozharskiy, Anton and Frey, Jonathan and Diehl, Moritz},
  journal={Nonlinear Analysis: Hybrid Systems},
  volume={54},
  pages={101518},
  year={2024},
  publisher={Elsevier}
}
~~~

#### Time-freezing

~~~bibtex
@article{Nurkanovic2021,
  title={A Time-Freezing Approach for Numerical Optimal Control of Nonsmooth Differential Equations with State Jumps},
  author={Nurkanovi{\'c}, Armin and Sartor, Thomas and Albrecht, Sebastian and Diehl, Moritz},
  journal={IEEE Control Systems Letters},
  year={2021}
}
~~~

~~~bibtex
@article{Nurkanovic2023,
  title={The Time-Freezing Reformulation for Numerical Optimal Control of Complementarity Lagrangian Systems with State Jumps},
  author={Nurkanovi{\'c}, Armin and Albrecht, Sebastian and Brogliato, Bernard and Diehl, Moritz},
  journal={Automatica},
  volume={158},
  pages={111295},
  year={2023}
}
~~~

~~~bibtex
@article{Nurkanovic2022a,
  title={Continuous Optimization for Control of Hybrid Systems with Hysteresis via Time-Freezing},
  author={Nurkanovi{\'c}, Armin and Diehl, Moritz},
  journal={IEEE Control Systems Letters},
  year={2022}
}
~~~

## Related software

### CCOpt

- [CCOpt.jl](https://github.com/MadNLP/CCOpt.jl)

~~~bibtex
@Article{Pozharskiy2026,
  Title                    = {{CCO}pt: an Open-Source Solver for Large-Scale Mathematical Programs with Complementarity Constraints},
  Author                   = {Pozharskiy, Anton and Pacaud, Fran{\c{c}}ois and Diehl, Moritz and Nurkanovi{\'c}, Armin},
  Journal                  = {arXiv preprint},
  Year                     = {2026},
  Url                      = {https://www.syscop.de/files/users/armin.nurkanovic/Pozharskiy2026.pdf}
}
~~~

### CasADi

- [CasADi -- A software framework for nonlinear optimization and optimal control](https://cdn.syscop.de/publications/Andersson2019.pdf)

### IPOPT

- [On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming](https://link.springer.com/article/10.1007/s10107-004-0559-y)

## Literature

### Real-time MPC algorithms

- [Real-Time Algorithms for Model Predictive Control of Hybrid Dynamical Systems](https://www.syscop.de/files/users/armin.nurkanovic/Nurkanovic2026a.pdf)

### FESD

- [Finite Elements with Switch Detection for Direct Optimal Control of Nonsmooth Systems](https://link.springer.com/article/10.1007/s00211-024-01412-z)
- [Finite Elements with Switch Detection for numerical optimal control of nonsmooth dynamical systems with set-valued heaviside step functions](https://www.sciencedirect.com/science/article/pii/S1751570X24000554)

### Projected dynamical systems 

- [First-Order Sweeping Processes and Extended Projected Dynamical Systems: Equivalence, Time-Discretization and Numerical Optimal Control](https://publications.syscop.de/Pozharskiy2025.pdf)
- [Finite Elements with Switch Detection for Numerical Optimal Control of Projected Dynamical Systems](https://publications.syscop.de/Pozharskiy2024c.pdf)


### Time-freezing

- [A Time-Freezing Approach for Numerical Optimal Control of Nonsmooth Differential Equations with State Jumps](https://cdn.syscop.de/publications/Nurkanovic2021.pdf)
- [The Time-Freezing Reformulation for Numerical Optimal Control of Complementarity Lagrangian Systems with State Jumps](https://www.sciencedirect.com/science/article/pii/S0005109823004594)
- [Continuous Optimization for Control of Hybrid Systems with Hysteresis via Time-Freezing](https://cdn.syscop.de/publications/Nurkanovic2022a.pdf)

## Contact

Questions, remarks, bug reports, and feature requests are best submitted via a new issue in this repository.

Main developers:
- Anton Pozharskiy — [anton.pozharskiy@imtek.uni-freiburg.de](mailto:anton.pozharskiy@imtek.uni-freiburg.de)
- Armin Nurkanović — [armin.nurkanovic@imtek.uni-freiburg.de](mailto:armin.nurkanovic@imtek.uni-freiburg.de)
- Jonathan Frey — [jonathan.frey@imtek.uni-freiburg.de](mailto:jonathan.frey@imtek.uni-freiburg.de)

Success stories and source code contributions are very welcome.