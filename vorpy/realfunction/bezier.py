"""
Module for working with Bezier curves.
"""

import numpy as np
import sympy as sp
import typing
import vorpy.realfunction.bernstein
import vorpy.symbolic

"""
Given boundary conditions up to order n (i.e. r := (r[0], ..., r[n]) and (s[0], ..., s[n]) which specify
the derivatives of the function up to the nth derivative at u = 0 and u = 1 respectively) for some n >= 0,
then want to find the control points for a Bezier curve of a particular degree fitting that discrete jet
function.

let c[k] be the kth control point which is to be solved for.  Could use the convention where the
choose(d,k) term is part of c[k].

    b(u) := sum(choose(d,k) * (1-u)**(d-k) * u**k * c[k])
          = sum(B(d,k)(u) * c[k]),

    where B(d,k) is the kth Bernstein polynomial of degree d.

then the boundary conditions are

    (d^j/du^j b)(0) = r[j] for j in 0,...,n
    (d^j/du^j b)(1) = s[j] for j in 0,...,n

This gives a linear relationship between c and (r,s).

Let the "Bezier tensor" be the linear transformation taking (r,s) to c.  This should be well-defined
if the degree of the Bezier parameterization is high enough.  This should be an invertible
transformation, because the derivatives of the Bezier parameterization can be computed.
"""

def bezier_tensor (*, degree:int, jet:int) -> np.ndarray:
    """
    Returns a tensor B[k,b,j] of shape (degree+1, 2, jet+1) defined by

        C[k] = B[k,b,j]*J[b,j] # summed over b and j

    where C is the Bezier coefficient tensor (its index is the same as the index
    for the corresponding Bernstein polynomial) and J is the jet tensor, where k
    indexes the coefficients of the Bernstein polynomial expression for the Bezier
    curve, where b indexes the t=0 or t=1 boundary, and j is the jet index (i.e.
    j=0 is 0th derivative, j=1 is 1st derivative, etc).

    TODO: This should specify the interval over which the Bernstein polynomials are defined.
    """

    if degree < 0:
        raise TypeError(f'expected degree (which was {degree}) to be nonnegative.')
    if jet < 0:
        raise TypeError(f'expected jet (which was {jet}) to be nonnegative.')

    c_t = vorpy.symbolic.tensor('c', (degree+1,))
    t = sp.var('t')
    #b = sum(vorpy.realfunction.bernstein.bernstein_polynomial(t, degree=degree, k=k) * c for k,c in zip(range(degree+1), c_t))
    b = np.dot(vorpy.realfunction.bernstein.bernstein_polynomial_basis(t, degree=degree), c_t)

    jet_t = vorpy.symbolic.tensor('j', (2,jet+1))

    equation_v = []
    for k in range(jet+1):
        b_diff_k = b.diff(t, k)
        equation_v.append(sp.Subs(b_diff_k, t, 0).doit() - jet_t[0,k])
        equation_v.append(sp.Subs(b_diff_k, t, 1).doit() - jet_t[1,k])

    solutions = sp.linsolve(equation_v, *c_t.tolist())
    if len(solutions) != 1:
        raise ValueError(f'ill-defined Bezier tensor (there were {len(solutions)} solutions, but expected exactly 1)')

    solution = np.array(list(solutions)[0])
    bezier_t = vorpy.symbolic.differential(solution, jet_t)
    assert bezier_t.shape == (degree+1,) + jet_t.shape
    return bezier_t

def cubic_interpolation (x_v:np.ndarray, y_jet_t:np.ndarray) -> typing.Callable[[typing.Any],typing.Any]:
    """
    Returns a function which evaluates the cubic Bezier interpolation of the given 1-jet function
    values (y_jet_t) at the given parameter values (x_v).

    x_v must have shape (T,), where T is the number of samples for the function.
    y_jet_t must have shape (T,j,L), where j (in {0,1}) indexes the jth jet (i.e. 0 = 0th deriv,
    1 = 1th deriv), and L is the tensor shape of the function value.

    x_v must be strictly increasing.  If an x value is plugged into the returned interpolator
    that is outside the range specified by x_v, then the appropriate edge segment (first or last)
    will be used, so that the interpolation just continues as a cubic Bezier off the ends.
    """

    if len(x_v.shape) != 1:
        raise TypeError(f'expected x_v to be a 1-tensor, but it was actually {len(x_v.shape)}')
    if len(y_jet_t.shape) < 2:
        raise TypeError(f'expected y_jet_t to be at least a 2-tensor, but it was actually {len(y_jet_t.shape)}')
    if x_v.shape[0] != y_jet_t.shape[0]:
        raise TypeError(f'expected 0th axis of y_jet_t to have same dimension as 0th (and only) axis of x_v, but they were {y_jet_t.shape[0]} and {x_v.shape[0]} respectively')
    if y_jet_t.shape[1] != 2:
        raise TypeError(f'expected 1th axis of y_jet_t to have dimension 2 (this is the jet axis)')
    if not np.all(x_v[:-1] < x_v[1:]):
        raise ValueError(f'expected x_v to be strictly increasing')

    x_t = np.copy(x_v)
    segment_length_v = np.diff(x_t)
    # Precompute the tensor that can be contracted with the appropriate Bernstein polynomial,
    # and also takes into account the non-unit-length segments over which the x_t values occur.
    # The indices here are y_t[s,k,L], where s is the segment index, k is the Bernstein
    # polynomial index (0 through 3), and L is the tensor multiindex for the "body" of y_jet_t.
    y_t = np.ndarray((len(segment_length_v),4)+y_jet_t.shape[2:], dtype=float)
    bezier_t = bezier_tensor(degree=3, jet=1).astype(float)
    for segment_index,(segment_length,x_a,x_b) in enumerate(zip(segment_length_v, x_t[:-1], x_t[1:])):
        # param_t is to scale the 1st deriv values to account for the non-unit-length
        # segments over which the piecewise parameterization is defined.
        param_t = np.diag([1.0, segment_length])
        y_t[segment_index,:,...] = np.einsum(
            'kbj,jJ,bJ...->k...',
            bezier_t,
            param_t,
            y_jet_t[segment_index:segment_index+2,...],
        )

    n = len(x_t)
    def interp (x:float):
        search_index = np.searchsorted(x_t, x, side='left')
        # NOTE: For now, assume cubic interpolation off the end segments, but linear
        # interpolation off the ends would also be reasonable.
        if search_index <= 0:
            search_index = 1
        elif search_index >= n:
            search_index = n-1
        assert 0 < search_index < n
        segment_index = search_index-1
        x_a = x_t[segment_index]
        assert x_t[segment_index+1] - x_a == segment_length_v[segment_index]
        t = (x - x_a) / segment_length_v[segment_index]
        bernstein_t = vorpy.realfunction.bernstein.bernstein_polynomial_basis(t, degree=3).astype(float)
        return np.einsum('k,k...', bernstein_t, y_t[segment_index,:,...])

    def interpolator (x:typing.Union[float,np.ndarray]) -> typing.Union[float,np.ndarray]:
        # There's probably a way smarter and more efficient way to do this.
        if isinstance(x, float):
            return interp(x)
        elif isinstance(x, np.ndarray):
            retval = np.ndarray(x.shape + y_jet_t.shape[2:], dtype=float)
            for I in vorpy.tensor.multiindex_iterator(x.shape):
                retval[I] = interp(x[I])
            return retval
        else:
            raise TypeError(f'Expected x to be a float or numpy.ndarray, but it was {type(x)}')

    return interpolator
