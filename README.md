# Vorpy

A Python package for (usually math-related) code that I've written that finds use over and over.

Vorpy is short for "VictOR dods' PYthon package".  Plus if you Google "vorpy", you come up with some pretty [hilarious](http://www.urbandictionary.com/define.php?term=Vorpy) [shit](http://vorpycrill.bandcamp.com/releases), so it seems like a good name choice.

# Contents

-   [vorpy](https://github.com/vdods/vorpy/tree/master/vorpy) : The root module for the vorpy package.
-   [vorpy.apply_along_axes](https://github.com/vdods/vorpy/blob/master/vorpy/apply_along_axes.py) : Multi-dimensional
    generalization of `numpy.apply_along_axis`.
-   [vorpy.pickle](https://github.com/vdods/vorpy/blob/master/vorpy/pickle.py) : A module which facilitates disk-based caching of
    data via pickle files.  Useful when using data that is time-consuming to generate but does not often change.
-   [vorpy.symbolic](https://github.com/vdods/vorpy/blob/master/vorpy/symbolic.py) : Symbolic calculus module.  This module eases
    use of `sympy`, facilitating the use of vector/tensor calculus (via `numpy.ndarray`s of symbols),  and which allows compilation
    of symbolic functions into Python code, caching the generated code if desired.  That process is known as 'lambdification' in
    `sympy`.  Take a look at the `cached_lambdified` function within this source.
-   [vorpy.symplectic_integration](https://github.com/vdods/vorpy/blob/master/vorpy/symplectic_integration/) : Module implementing
    symplectic integrators; available is a family of separable Hamiltonian integrators and a nonseparable Hamiltonian integrator.
-   [vorpy.tensor](https://github.com/vdods/vorpy/blob/master/vorpy/tensor.py) : Module providing functions for working with tensor values, notably the `contract` function, which does the same thing as `numpy.einsum` but allows `dtype=object`.

# How to Install

Vorpy can be installed via `pip` using the following command:

    pip install vorpy

Or, it can be installed directly from this github.com repository using the following command:

    pip install --upgrade git+https://github.com/vdods/vorpy.git

Installation can also be made from a local copy of the vorpy package repo:

    pip install --upgrade path/to/vorpy

where path/to/vorpy is the root directory of the project; the path containing setup.py (as well as this README.md file).  Apropos: see `pip install --editable <path/url>`

To uninstall, use the following obvious command:

    pip uninstall vorpy

A tarball of a particular tagged version can be obtained from github using a particular link.  The latest version, with
git tag `v0.4.3`, can be downloaded from:

    https://github.com/vdods/vorpy/archive/v0.4.3.tar.gz

Extracting this will create and populate the `vorpy-0.4.3` directory.

# Running Tests

The suite of unit tests can be run via the command:

    python setup.py test

# Release Notes

- `0.7.0` : Modification to the symbolic polynomial generating functions.
- `0.6.0` : Added the `vorpy.manifold` and `vorpy.symplectic` modules, which implement some basic calculus in those categories.
- `0.5.1` : Fixed a bug where the Python 2 `reduce` was moved to the `functools` module, and so caused a NameError in Python 3.
- `0.5.0` : Added `vorpy.tensor`.
- `0.4.3` : More updates to be able to submit to `pypi`.  Made tests generate files to the `test_artifacts` directory.
- `0.4.2` : Updates to be able to submit to `pypi` -- the Python Package Index.  Also a minor exception printing improvement.
- `0.4.1` : Changed `vorpy.pickle` to use the `dill` module (which can pickle lambda expressions) instead of the builtin `pickle` module.
- `0.4.0` : Added `vorpy.pickle`.
- `0.3.1` : Added a means to salvage results from symplectic integration if an exception is raised during integration.
- `0.3.0` : Added `vorpy.symplectic_integration.nonseparable_hamiltonian`.
- `0.2.0` : Added `vorpy.symplectic_integration.separable_hamiltonian`.
- `0.1.0` : Added `vorpy.apply_along_axes`.
- `0.0.0` : Initial release.  Added `vorpy.symbolic`.

# To-Do List

-   Consolidate vorpy.symbolic.multiindex_iterator and vorpy.tensor.multiindex_iterator (probably use the one in
    symbolic because it looks fancier, but it does belong within the vorpy.tensor module).
-   Include tests in the dist package, so they can be tested at the install site.
-   Make the `symbolic` module aware of vectorized operations so that fast `numpy`-implemented `ndarray` functions
    can be used instead of structure-forgetting symbolic expressions that are fully written out.  For example,
    the 1st and 2nd total derivatives of a quadratic form are simply matrix expressions which have simple `numpy`
    expressions.
-   Verify that `vorpy.apply_along_axes` is automatically parallelized.
-   Require `numpy.ndarray` or `tuple` be the type of the input array(s) for `apply_along_axes`, so that extra parens
    to form a trivial tuple are not necessary in the basic case where there is a single input array.
-   In `apply_along_axes`, allow the base case where `input_array_v[0].shape == ()` and `input_axis_v == []`.
-   Increasing the order of Tao's nonseparable Hamiltonian integrator doesn't actually do what it should, even when
    using very small timesteps (expected behavior is that the error is orders smaller, but what actually happens is
    that it's of the same order as say order=2).  Examine why this is happening (perhaps published version of Tao
    paper is updated with correction).
-   Create a [symplectic] integrator using the [Jacobi-Maupertuis principle](https://en.wikipedia.org/wiki/Maupertuis%27_principle).
-   Specify an upper bound for integrator on H error, which abs(H - H(qp(0))).  When computing the next step, if the
    H error is above the upper bound, recompute that step with a lower dt.  This will involve defining a search
    strategy for dt that doesn't make it too small.
-   Move `PendulumNd` and `KeplerNd` from tests dir into vorpy and write tests to symbolically verify all the formulas
    are correct.  The goal would be to create more of these to provide a whole family of dynamical systems that can
    be used and experimented with.
-   Have tests create a `test_artifacts` subdir which all files that the tests produce are put in, so there's an
    easy single thing to delete after tests run (or tests can clean this up themselves).
-   Switch to [pytest](https://docs.pytest.org/en/latest/).
-   For `sys.stderr.write`-based warnings, create a flag that can be set within the `vorpy` module to silence them.
-   Use `hashlib` for data caching layer on top of `vorpy.pickle`.
-   Make a human-readable hash, which concatenates words to represent elements of a hash space.  This should be
    convertable to/from a binary string.  Examples:

        StupidlyWalkingPerson
        BigThing
        Tree

    Maybe also tack on the remaining bits to the end as a decimal or hex number.  Examples:

        StupidlyWalkingPersonA07BE5
        Tree10034

    See [this](https://stackoverflow.com/questions/17753182/getting-a-large-list-of-nouns-or-adjectives-in-python-with-nltk-or-python-mad).
-   Use hashing in `vorpy.symbolic.cached_lambdified` on what gets lambdified so that changes to the function automatically
    cause the cache to be updated.
-   Make a module that provides a nice abstraction for concurrent.futures.ProcessPoolExecutor for parallel processing.
-   Allow multiply-nested dirs for cache_dirname in cached_lambdified -- this would correspond to multiply-nested modules
    for the cached code.  Use os.makedirs instead of os.mkdir

# References

-   [Maintainer Release Instructions](release-instructions.md)
-   [http://docs.python-guide.org/en/latest/writing/structure/]
-   [https://en.wikipedia.org/wiki/Symplectic_integrator]
-   [https://en.wikipedia.org/wiki/Energy_drift]
-   [https://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.043303] - Molei Tao - Explicit symplectic approximation
    of nonseparable Hamiltonians: Algorithm and long time performance
-   The package index for [dill](https://pypi.python.org/pypi/dill) (which is imported by `vorpy.pickle`) makes the following request:

        If you use dill to do research that leads to publication, we ask that you acknowledge use of dill by citing the following in your publication:

            M.M. McKerns, L. Strand, T. Sullivan, A. Fang, M.A.G. Aivazis,
            "Building a framework for predictive science", Proceedings of
            the 10th Python in Science Conference, 2011;
            http://arxiv.org/pdf/1202.1056

            Michael McKerns and Michael Aivazis,
            "pathos: a framework for heterogeneous computing", 2010- ;
            http://trac.mystic.cacr.caltech.edu/project/pathos

        Please see http://trac.mystic.cacr.caltech.edu/project/pathos or http://arxiv.org/pdf/1202.1056 for further information.
