# `nosnoc`


## About
`nosnoc` is an open source Python software package for NOnSmooth Numerical Optimal Control.

It is a structured reimplementation of the Matlab package NOSNOC (https://github.com/nurkanovic/nosnoc), but does not support all features in there (yet).
Most importantly, time freezing is not implemented yet.

It implements the FESD (Finite elements with switch detection) method, which allows one to handle nonsmooth ODE systems accurately in simulation and optimal control problems.

More information can be found in the NOSNOC package (https://github.com/nurkanovic/nosnoc).


## Installation

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
pip install -e .
```

4. Docs
In order to build docs also run:
```
pip -r docs_reqirements.txt
```

## Literature - theory and algorithms

### FESD
[Finite Elements with Switch Detection for Direct Optimal Control of Nonsmooth Systems](https://arxiv.org/abs/2205.05337) \
A.Nurkanović, M. Sperl, S. Albrecht, M. Diehl \
arXiv preprint 2022


### NOSNOC

[NOSNOC: A Software Package for Numerical Optimal Control of Nonsmooth Systems](https://cdn.syscop.de/publications/Nurkanovic2022b.pdf) \
A.Nurkanović , M. Diehl \
IEEE Control Systems Letters 2022


<!-- TODO: uncomment this when time freezing is implemplemented -->
<!--
### Time - Freezing
[A Time-Freezing Approach for Numerical Optimal Control of Nonsmooth Differential Equations with State Jumps](https://cdn.syscop.de/publications/Nurkanovic2021.pdf) \
A. Nurkanović, T. Sartor, S. Albrecht, M. Diehl \
IEEE Control Systems Letters 2021

[The Time-Freezing Reformulation for Numerical Optimal Control of Complementarity Lagrangian Systems with State Jumps](https://arxiv.org/abs/2111.06759) \
A. Nurkanović, S. Albrecht, B. Brogliato, M. Diehl \
arXiv preprint 2021

[Continuous Optimization for Control of Hybrid Systems with Hysteresis via Time-Freezing](https://cdn.syscop.de/publications/Nurkanovic2022a.pdf) \
A.Nurkanović , M. Diehl \
IEEE Control Systems Letters 2022 -->
