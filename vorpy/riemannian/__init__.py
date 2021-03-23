import numpy as np
import sympy as sp
#import typing
import vorpy.symbolic
import vorpy.tensor

# TODO: Maybe create a Riemannian coordinate chart, which is a set of coordinates,
# a metric tensor field, and an inverse metric tensor field.  Probably should incorporate
# vorpy.experimental.coordinates into this.

def christoffel_symbol (g:np.ndarray, g_inv:np.ndarray, X:np.ndarray):
    """
    The Christoffel symbol(s) Gamma^{i}_{jk} are a correction to the coordinate-dependent total derivative
    to produce the metric-compatible covariant derivative (Levi-Civita).  Note that while Gamma is a
    3-dimensional matrix, it isn't a tensorial value, since it's coordinate dependent.

    For now, assume g has shape (n,n) for some integer n.  g_inv is an explicit parameter
    because inverting a matrix is nontrivial and
    TODO: Allow arbitrary g having shape s+s for some shape s.
    """

    s = np.shape(g)
    assert len(s) == 2, 'g must be a 2-tensor'
    assert s[0] == s[1], 'g must be square'

    dg = vorpy.symbolic.differential(g, X)
    # TODO: See about forming the sum g_{jl,k} + g_{kl,j} - g_{jk,l} beforehand.
    christoffel = sp.Rational(1,2)*(
          vorpy.tensor.contract('il,jlk', g_inv, dg)
        + vorpy.tensor.contract('il,klj', g_inv, dg)
        - vorpy.tensor.contract('il,jkl', g_inv, dg)
    )
    return sp.simplify(christoffel)

def covariant_derivative_of (V:np.ndarray, Gamma:np.ndarray, X:np.ndarray) -> np.ndarray:
    return vorpy.symbolic.differential(V, X) + vorpy.tensor.contract('ijk,j', Gamma, V)

if __name__ == '__main__':
    # Test some stuff

    r, theta = sp.var('r'), sp.var('theta')
    print(f'r = {r}')
    print(f'r = {theta}')

