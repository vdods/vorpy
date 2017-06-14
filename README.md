# Vorpy

A Python package for (usually math-related) code that I've written that finds use over and over.

Vorpy is short for "VictOR dods' PYthon package".  Plus if you Google "vorpy", you come up with some pretty [hilarious](http://www.urbandictionary.com/define.php?term=Vorpy) [shit](http://vorpycrill.bandcamp.com/releases), so it seems like a pretty good name choice.

# Contents

| Module | Description |
| ------ | ----------- |
| [vorpy](https://github.com/vdods/vorpy/tree/master/vorpy) | The root module for the vorpy package. |
| [vorpy.apply_along_axes](https://github.com/vdods/vorpy/blob/master/vorpy/apply_along_axes.py) | Multi-dimensional generalization of `numpy.apply_along_axis`. |
| [vorpy.symbolic](https://github.com/vdods/vorpy/blob/master/vorpy/symbolic.py) | Symbolic calculus module.  This module eases use of `sympy`, facilitating the use of vector/tensor calculus (via `numpy.ndarray`s of symbols),  and which allows compilation of symbolic functions into Python code, caching the generated code if desired.  That process is known as 'lambdification' in `sympy`.  Take a look at the `cached_lambdified` function within this source. |
| [vorpy.symplectic_integration](https://github.com/vdods/vorpy/blob/master/vorpy/symplectic_integration/) | Module implementing symplectic integrators; only currently available type is a family of split-Hamiltonian integrators. |

# How to Install

Vorpy can be installed directly from this github.com repository using the following command:

    pip install git+https://github.com/vdods/vorpy.git

Installation can also be made from a local copy of the vorpy package repo:

    pip install path/to/vorpy

where path/to/vorpy is the root directory of the project; the path containing setup.py (as well as this README.md file).  Apropos: see `pip install --editable <path/url>`

To uninstall, use the following obvious command:

    pip uninstall vorpy

# Running Tests

The suite of unit tests can be run via the command:

    python setup.py test

# Release Notes

| Version | Notes |
| ------- | ----- |
| 0.0.0 | Initial release.  Added `vorpy.symbolic`. |
| 0.1.0 | Added `vorpy.apply_along_axes`. |
| 0.2.0 | Added `vorpy.symplectic_integration`. |

# References

-   [http://docs.python-guide.org/en/latest/writing/structure/]

# To-Do List

-   Verify that `vorpy.apply_along_axes` is automatically parallelized.
