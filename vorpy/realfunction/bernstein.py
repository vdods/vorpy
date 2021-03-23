"""
Module for working with Bernstein polynomials.
"""

import numpy as np
import sympy as sp
import typing
import vorpy.symbolic

def bernstein_polynomial (t:typing.Any, *, degree:int, k:int) -> typing.Any:
    """
    Returns the kth Bernstein polynomial of given degree in the given symbol.
    It's assumed that the interval for the Bernstein polynomial is [0,1].

    TODO: This should also specify the interval over which the polynomial is defined.
    TODO: Should be able to specify dtype.
    """
    if degree < 0:
        raise TypeError(f'expected degree (which was {degree}) to be nonnegative.')
    if k < 0 or k > degree:
        raise ValueError(f'expected 0 <= k <= degree (but k = {k} and degree = {degree})')

    s = 1-t
    return sp.binomial(degree,k) * s**(degree-k) * t**k

def bernstein_polynomial_basis (t:typing.Any, *, degree:int) -> np.ndarray:
    """
    Returns a vector of Bernstein polynomials of the specified degree with k = 0 to degree.
    If t is a symbol, then this is a symbolic vector.  If t is a float, then this is a float
    valued vector.

    TODO: This should also specify the interval over which the polynomial is defined.
    TODO: Should be able to specify dtype.
    """
    return np.array([bernstein_polynomial(t, degree=degree, k=k) for k in range(degree+1)])

def bernstein_polynomial_L2_inner_product (t:typing.Any, *, degree:int) -> np.ndarray:
    B = bernstein_polynomial_basis(t, degree=degree)

    def L2_inner_product (f0, f1) -> typing.Any:
        return sp.integrate(f0*f1, (t, 0, 1))

    return np.array([
        [L2_inner_product(b_i,b_j) for b_j in B]
        for b_i in B
    ])

def bernstein_polynomial_L2_inner_product_inv (t:typing.Any, *, degree:int) -> np.ndarray:
    B = bernstein_polynomial_basis(t, degree=degree)

    def L2_inner_product (f0, f1) -> typing.Any:
        return sp.integrate(f0*f1, (t, 0, 1))

    L2_in_B = bernstein_polynomial_L2_inner_product(t, degree=degree)
    return np.array(sp.Matrix(L2_in_B).inv().tolist())
