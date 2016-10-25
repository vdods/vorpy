# vorpy

A Python package for (usually math-related) code that I've written that finds use over and over.

Vorpy is short for "VictOR dods' PYthon package".  Plus, if you Google "vorpy", you come up with some pretty [hilarious](http://www.urbandictionary.com/define.php?term=Vorpy) [shit](http://vorpycrill.bandcamp.com/releases).

# Contents

-   `symbolic.py` : A module which eases use of `sympy`, facilitating the use of vector/tensor calculus (via `numpy.ndarray`s of symbols),  and which allows compilation of symbolic functions into Python code, caching the generated code if desired.  That process is known as 'lambdification' in `sympy`.  Take a look at the `cached_lambdified` function within this source.
-   Other stuff to be listed later; `symbolic.py` is the gem of this package.

# How to install via `pip`

Vorpy can be installed directly from this github.com repository using the following command:

    pip install git+https://github.com/vdods/vorpy.git

Installation can also be made from a local copy of the vorpy package repo:

    pip install path/to/vorpy

where path/to/vorpy is the root directory of the project; the path containing setup.py (as well as this README.md file).  Apropos: see `pip install -e <url>`

To uninstall, use the following obvious command:

    pip uninstall vorpy

# References

-   [http://docs.python-guide.org/en/latest/writing/structure/]
