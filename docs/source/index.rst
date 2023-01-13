Welcome to nosnoc's documentation!
===================================

About
-----

`nosnoc` is an open source Python software package for NOnSmooth Numerical Optimal Control.

It is a structured reimplementation of the Matlab package NOSNOC (https://github.com/nurkanovic/nosnoc), but does not support all features in there (yet).
Most importantly, time freezing is not implemented yet.

It implements the FESD (Finite elements with switch detection) method, which allows one to handle nonsmooth ODE systems accurately in simulation and optimal control problems.

More information can be found in the NOSNOC package (https://github.com/nurkanovic/nosnoc).


.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
   api
   installation
