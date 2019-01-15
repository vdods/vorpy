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

def bernstein_embedding (*, source_degree:int, dest_degree:int) -> np.ndarray:
    """
    NOTE: Not sure if this is totally right -- need to implement correct test

    Returns the linear map taking the space of source_degree-degree Bernstein polynomials to the space
    of dest_degree Bernstein polynomials.  The return value is a matrix having shape
    (dest_degree+1, source_degree+1).

    dest_degree must be greater or equal to source_degree.
    """

    if source_degree > dest_degree:
        raise ValueError(f'expected source_degree (which was {source_degree}) < dest_degree (which was {dest_degree}')

    c_t         = vorpy.symbolic.tensor('c', (source_degree+1,))
    t           = sp.var('t')
    b_source    = np.dot(bernstein_polynomial_basis(t, degree=source_degree), c_t)

    # Use the L2 inner product <<x,y>> on the interval [0,1] to find the embedding.  In particular,
    #
    #     E[r,c] = << B(dest_degree,c), B(source_degree,r) >>,
    #
    # B(d,k) denotes the kth Bernstein polynomial of degree d.

    def L2_inner_product (f0, f1) -> typing.Any:
        return sp.integrate(f0*f1, (t, 0, 1))

    B_source = bernstein_polynomial_basis(t, degree=source_degree)
    B_dest = bernstein_polynomial_basis(t, degree=dest_degree)

    return np.dot(
        np.array([
            [L2_inner_product(b_source,b_dest) for b_source in B_source]
            for b_dest in B_dest
        ]),
        bernstein_polynomial_L2_inner_product_inv(t, degree=source_degree),
    )

if __name__ == '__main__':
    import sys

    t = sp.var('t')

    bernstein_polynomial_L2_inner_product_inv(t, degree=2)

    BE = bernstein_embedding(source_degree=2, dest_degree=3)
    print(BE)
    print(np.dot(np.dot(BE.T, bernstein_polynomial_L2_inner_product(t, degree=3)), BE))
    print('TODO: Fix this')

    sys.exit(0)

