import numpy as np
import scipy.interpolate
import typing

# TODO: Could define the orientation of zeros to be the sign of the [symmetric] discrete
# derivative, which is a reasonable thing to do if the interpolation scheme is cubic Bezier.

def oriented_zeros (
    f_v:np.ndarray,
    *,
    t_v:typing.Optional[np.ndarray]=None,
    orientation_p:typing.Callable[[int],bool]=(lambda o:True),
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the zeros of the the piecewise-linear function specified by the discretization

        t_v[i] |-> f_v[i].

    A zero at time t_z has positive orientation if

        f(t) <= 0 for t_z-epsilon < t < t_z  and  f(t) > 0 for t_z < t < t_z+epsilon.

    A zero at time t_z has negative orientation if

        f(t) >= 0 for t_z-epsilon < t < t_z  and  f(t) < 0 for t_z < t < t_z+epsilon.

    A zero at time t_z has zero orientation if

        f(t) >= 0 for t_z-epsilon < t < t_z+epsilon  or  f(t) <= 0 for t_z-epsilon < t < t_z+epsilon.

    Returns zero_index_pair_v, zero_orientation_v, zero_t_v

    t_v may be used to specify the times (domain values) corresponding the the elements of f_v.
    If t_v is not specified, then it is taken to be np.arange(len(f_v)).

    orientation_p is a predicate defining the condition to include a particular zero based on its orientation.
    The default predicate is the constant `True` function (i.e. include all zeros).

    Only returns nondegenerate zeros (i.e. if there are any sequences of repeated zeros in f_v, then only the
    boundary zeros for each sequence are returned).
    """

    n = len(f_v)

    if t_v is None:
        t_v = np.arange(n)
    else:
        if len(t_v) != n:
            raise TypeError(f'expected len(f_v) == len(t_v), but len(f_v) = {len(f_v)} and len(t_v) = {len(t_v)}')

    if n < 2:
        # Degenerate case.
        zero_index_pair_v = np.ndarray((0,1), dtype=int)
        zero_orientation_v = np.array([], dtype=int)
        zero_t_v = np.array([], dtype=t_v.dtype)
        return zero_index_pair_v, zero_orientation_v, zero_t_v

    sign_f_v = np.sign(f_v)

    # Handle exact zeros

    twosided_exact_zero_index_v = np.where((f_v[1:-1] == 0) & (f_v[:-2] != 0) & (f_v[2:] != 0))[0]+1
    # right_exact_zero_index_v should not include twosided_exact_zero_index_v elements.
    right_exact_zero_index_v = np.array(sorted(frozenset(np.where((f_v[:-1] == 0) & (f_v[1:] != 0))[0]).difference(twosided_exact_zero_index_v)), dtype=int)
    # left_exact_zero_index_v should not include twosided_exact_zero_index_v or left_exact_zero_index_v elements.
    left_exact_zero_index_v  = np.array(sorted(frozenset(np.where((f_v[:-1] != 0) & (f_v[1:] == 0))[0] + 1).difference(twosided_exact_zero_index_v, right_exact_zero_index_v)), dtype=int)

    assert np.all(twosided_exact_zero_index_v > 0)
    assert np.all(twosided_exact_zero_index_v < n-1)
    assert np.all(right_exact_zero_index_v    < n-1)
    assert np.all(left_exact_zero_index_v     > 0)

    twosided_exact_zero_orientation_v = np.sign(sign_f_v[twosided_exact_zero_index_v+1] - sign_f_v[twosided_exact_zero_index_v-1])
    right_exact_zero_orientation_v = sign_f_v[right_exact_zero_index_v+1]
    left_exact_zero_orientation_v = -sign_f_v[left_exact_zero_index_v-1]

    # Filter by orientation

    filter_index_v = [i for i,o in enumerate(twosided_exact_zero_orientation_v) if orientation_p(o)]
    twosided_exact_zero_index_v = twosided_exact_zero_index_v[filter_index_v]
    twosided_exact_zero_orientation_v = twosided_exact_zero_orientation_v[filter_index_v]

    filter_index_v = [i for i,o in enumerate(right_exact_zero_orientation_v) if orientation_p(o)]
    right_exact_zero_index_v = right_exact_zero_index_v[filter_index_v]
    right_exact_zero_orientation_v = right_exact_zero_orientation_v[filter_index_v]

    filter_index_v = [i for i,o in enumerate(left_exact_zero_orientation_v) if orientation_p(o)]
    left_exact_zero_index_v = left_exact_zero_index_v[filter_index_v]
    left_exact_zero_orientation_v = left_exact_zero_orientation_v[filter_index_v]

    a = len(twosided_exact_zero_index_v)
    b = a + len(right_exact_zero_index_v)
    exact_zero_count = b + len(left_exact_zero_index_v)

    # Handle inexact zeros

    inexact_zero_index_v = np.where((sign_f_v[:-1] != sign_f_v[1:]) & (sign_f_v[:-1] != 0) & (sign_f_v[1:] != 0))[0]

    inexact_zero_orientation_v = sign_f_v[inexact_zero_index_v+1]

    assert np.all(np.sign(f_v[inexact_zero_index_v+1]) != np.sign(f_v[inexact_zero_index_v]))

    # Filter by orientation

    filter_index_v = [i for i,o in enumerate(inexact_zero_orientation_v) if orientation_p(o)]
    inexact_zero_index_v = inexact_zero_index_v[filter_index_v]
    inexact_zero_orientation_v = inexact_zero_orientation_v[filter_index_v]

    inexact_zero_count = len(inexact_zero_index_v)

    # Put it all together

    zero_count = exact_zero_count + inexact_zero_count

    zero_index_pair_v = np.ndarray((zero_count,2), dtype=right_exact_zero_index_v.dtype)
    zero_orientation_v = np.ndarray((zero_count,), dtype=sign_f_v.dtype)

    # This may be no elements, but numpy's semantics handle that seamlessly

    zero_index_pair_v[:a,0] = twosided_exact_zero_index_v
    zero_index_pair_v[:a,1] = twosided_exact_zero_index_v+1
    zero_orientation_v[:a] = twosided_exact_zero_orientation_v

    zero_index_pair_v[a:b,0] = right_exact_zero_index_v
    zero_index_pair_v[a:b,1] = right_exact_zero_index_v+1
    zero_orientation_v[a:b] = right_exact_zero_orientation_v

    zero_index_pair_v[b:exact_zero_count,0] = left_exact_zero_index_v-1
    zero_index_pair_v[b:exact_zero_count,1] = left_exact_zero_index_v
    zero_orientation_v[b:exact_zero_count] = left_exact_zero_orientation_v

    zero_index_pair_v[exact_zero_count:,0] = inexact_zero_index_v
    zero_index_pair_v[exact_zero_count:,1] = inexact_zero_index_v+1
    zero_orientation_v[exact_zero_count:] = inexact_zero_orientation_v

    # Now sort it

    sorted_index_v = np.argsort(zero_index_pair_v[:,0])

    zero_index_pair_v[...] = zero_index_pair_v[sorted_index_v,:]
    zero_orientation_v[...] = zero_orientation_v[sorted_index_v]

    assert np.all(zero_index_pair_v[:,0] + 1 == zero_index_pair_v[:,1])
    assert np.all(sign_f_v[zero_index_pair_v[:,0]] != sign_f_v[zero_index_pair_v[:,1]])

    # Compute the zero times.

    t_delta_v = t_v[zero_index_pair_v[:,1]] - t_v[zero_index_pair_v[:,0]]
    f_delta_v = f_v[zero_index_pair_v[:,1]] - f_v[zero_index_pair_v[:,0]]
    assert np.all(f_delta_v != 0)
    zero_t_v = t_v[zero_index_pair_v[:,0]] - f_v[zero_index_pair_v[:,0]]*t_delta_v/f_delta_v

    assert np.allclose(np.interp(zero_t_v, t_v, f_v), 0)

    return zero_index_pair_v, zero_orientation_v, zero_t_v

#def zeros (f_v:np.ndarray, *, orientation_vo:typing.Sequence[int]=None) -> np.ndarray:
    #"""
    #Returns zeros of the piecewise-linear, real function t |-> f(t) with t in [0, n-1], where n = len(f_v).
    #In particular, the function is defined is the piecewise-linear interpolation of the discrete function
    #i |-> f_v[i] for i in [0,1,...,n-1].

    #The orientation parameter may be used to specify additional filtering of zeros based on their orientation.
    #The orientation of a zero f(t_z) == 0 is as follows.

    #-   Positive orientation:
        #-   If f(t) < 0 for t_z-epsilon < t < t_z and f(t) > 0 for t_z < t < t_z+epsilon, then the orientation
            #of the zero is positive (i.e. the function becomes positive).
        #-   If t_z == t_v[0] and f(t) > 0 for t_z < t < t_z+epsilon, then the orientation of the zero is positive
            #(i.e. the function starts at zero at the domain's lower boundary and becomes positive).
        #-   If t_z == t_v[n-1] and f(t) < 0 for t_z-epsilon < t < t_z, then the orientation of the zero is positive.
            #(i.e. the function becomes zero at the domain's upper boundary having previously been negative).
    #-   Negative orientation:
        #-   If f(t) > 0 for t_z-epsilon < t < t_z and f(t) < 0 for t_z < t < t_z+epsilon, then the orientation
            #of the zero is negative (i.e. the function becomes negative).
        #-   If t_z == t_v[0] and f(t) < 0 for t_z < t < t_z+epsilon, then the orientation of the zero is negative
            #(i.e. the function starts at zero at the domain's lower boundary and becomes negative).
        #-   If t_z == t_v[n-1] and f(t) > 0 for t_z-epsilon < t < t_z, then the orientation of the zero is negative.
            #(i.e. the function becomes zero at the domain's upper boundary having previously been positive).
    #-   Zero orientation:
        #-   If

    #-

    #-   orientation == 0 : return all zeros
    #-   orientation < 0  : return negatively oriented zero crossings (where the function goes from positive to negative)
    #-   orientation > 0  : return positively oriented zero crossings (where the function goes from negative to positive)
    #"""

    #if len(t_v) != len(f_v):
        #raise TypeError(f'expected len(t_v) == len(f_v), but got len(t_v) = {len(t_v)} and len(f_v) = {len(f_v)}')

    ## zc stands for zero crossing.

    ## Non-positive elements of this indicate a zero crossing.
    #zc_discriminant_v = f_v[:-1] * f_v[1:]
    ## Consider only strictly negative discriminant as indicating a zero crossing.  This will not pick up
    ## cases where there is a repeated zero, or where the function touches but doesn't cross zero.
    #zc_v = zc_discriminant_v < 0
    #zc_index_v = np.where(zc_v)[0]
    #assert np.all(zc_index_v < len(t_v)-1)
    #if orientation != 0:
        #zc_orientation_v = np.sign(f_v[zc_index_v+1] - f_v[zc_index_v])
        #assert np.all(zc_orientation_v != 0), 'this should be true by construction (following the zc_discriminant_v < 0 condition)'
        #zc_index_v = zc_index_v[zc_orientation_v == np.sign(orientation)]

    #assert np.all(np.sign(f_v[zc_index_v+1]) != np.sign(f_v[zc_index_v]))
    #assert np.all(f_v[zc_index_v+1]*f_v[zc_index_v] < 0), 'this should be equivalent to the sign check, but is done using discriminant'
    #if orientation != 0:
        #assert np.all(np.sign(f_v[zc_index_v+1]) == np.sign(orientation))

    #zc_index_pair_t = np.ndarray((len(zc_index_v),2), dtype=int)
    #zc_index_pair_t[:,0] = zc_index_v
    #zc_index_pair_t[:,1] = zc_index_v+1
    #assert np.all(zc_index_pair_t < len(t_v)), 'each element of zc_index_pair_t should be a valid index for both t_v and f_v'

    ## Make tensors quantifying the intervals containing the zero crossings.
    ## Note here that because zc_index_pair_t is a 2-tensor, and t_v and f_v are 1-tensors,
    ## zc_interval_t_v and zc_interval_f_v will be a 2-tensor whose rows are the interval bounds.
    #zc_interval_t_v = t_v[zc_index_pair_t]
    #zc_interval_f_v = f_v[zc_index_pair_t]
    #assert zc_interval_t_v.shape == (len(zc_index_v),2)
    #assert zc_interval_f_v.shape == (len(zc_index_v),2)

    ## For each zero crossing, use a piecewise linear interpolation of f_v to solve for a better
    ## approximation of the exact time it crosses zero.
    #zc_t_delta_v = np.diff(zc_interval_t_v, axis=1).reshape(-1)
    #zc_f_delta_v = np.diff(zc_interval_f_v, axis=1).reshape(-1)
    #zc_t_v = zc_interval_t_v[:,0] - zc_interval_f_v[:,0]*zc_t_delta_v/zc_f_delta_v

    ### Numerical sanity check (the bound is based on the max number encountered in the solution for the respective component of zc_t_v).
    ##assert np.all(np.interp(zc_t_v, t_v, f_v) < 1.0e-8*np.max(zc_interval_f_v, axis=1))

    ## Add the endpoint zeros if requested.
    #if include_endpoint_zeros:


    #return zc_t_v, zc_index_pair_t

def critical_points (
    f_v:np.ndarray,
    *,
    t_v:typing.Optional[np.ndarray]=None,
    orientation_p:typing.Callable[[int],bool]=(lambda o:True),
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns critical points of the function f_v (i.e. where its discrete derivative is zero).  The indices
    will be in the range [1, n-2], where n == len(f_v).  The derivative defining the critical point condition
    is the symmetric derivative defined as the slope of the secant line joining the left and right neighboring
    points in the function's discrete representation.

    The orientation of a critical point is positive if the critical point is a minimum, and negative if the
    critical point is a maximum (the orientation can be thought of as the sign of the 2nd derivative of the
    function).

    NOTE: This function technically shouldn't be in piecewiselinear, as it's treating the derivative of the
    function as piecewise linear.  This could be considered to be a piecewise quadratic Bezier interpolation
    scheme.
    """

    n = len(f_v)

    if t_v is None:
        t_v = np.arange(n)
    else:
        if len(t_v) != n:
            raise TypeError(f'expected len(f_v) == len(t_v), but len(f_v) = {len(f_v)} and len(t_v) = {len(t_v)}')

    if len(f_v) < 3:
        # Degenerate case
        critical_point_index_pair_v = np.ndarray((0,1), dtype=int)
        critical_point_orientation_v = np.array([], dtype=int)
        critical_point_t_v = np.array([], dtype=t_v.dtype)
        return critical_point_index_pair_v, critical_point_orientation_v, critical_point_t_v

    # Use a symmetric definition of derivative.
    discrete_deriv_f_v = (f_v[2:] - f_v[:-2]) / (t_v[2:] - t_v[:-2])
    critical_point_index_pair_v, critical_point_orientation_v, critical_point_t_v = oriented_zeros(discrete_deriv_f_v, t_v=t_v[1:-1], orientation_p=orientation_p)

    # Adjust the index to match the original f_v
    critical_point_index_pair_v += 1

    return critical_point_index_pair_v, critical_point_orientation_v, critical_point_t_v

def local_maximizers (
    f_v:np.ndarray,
    *,
    t_v:typing.Optional[np.ndarray]=None,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NOTE: This function, like critical_points, technically doesn't belong in piecewiselinear.
    """
    return critical_points(f_v, t_v=t_v, orientation_p=(lambda o:o < 0))

def local_minimizers (
    f_v:np.ndarray,
    *,
    t_v:typing.Optional[np.ndarray]=None,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NOTE: This function, like critical_points, technically doesn't belong in piecewiselinear.
    """
    return critical_points(f_v, t_v=t_v, orientation_p=(lambda o:o > 0))

#def local_maximizers (
    #f_v:np.ndarray,
    #*,
    #t_v:typing.Optional[np.ndarray]=None,
#) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #"""
    #Returns maximizer_index_v, maximizer_f_v, maximizer_t_v, each having the same length, where

        #maximizer_t_v[maximizer_index_v[i]] |-> maximizer_f_v[maximizer_index_v[i]]

    #is a strict local max.  If t_v is not specified, then t_v=np.arange(len(f_v)), and so maximizer_t_v
    #will be the same as maximizer_index_v.
    #"""

    #n = len(f_v)

    #if n == 0:
        #raise ValueError(f'can not compute local_maximizers of empty array f_v')

    #if t_v is None:
        #t_v = np.arange(n)
    #else:
        #if len(t_v) != n:
            #raise TypeError(f'expected len(f_v) == len(t_v), but len(f_v) = {len(f_v)} and len(t_v) = {len(t_v)}')

    #maximizer_index_v = []
    #maximizer_f_v = []

    #def is_local_max (i:int) -> bool:
        #return (i > 0 and

    #for i in range(len(f_v))


    #return

#def local_minimizers (
    #f_v:np.ndarray,
    #*,
    #t_v:typing.Optional[np.ndarray]=None,
#) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ## TODO: similar to local_maximizers

if __name__ == '__main__':
    f_v = np.array([0, 2, 0, 2, -2, 2, 2, 0, 0, 0, -2, -2, -3, 0, -3, 0, 0, -3, 0])
    t_v = np.arange(len(f_v))
    zero_index_pair_v, zero_orientation_v, zero_t_v = oriented_zeros(f_v)
    print(f'zero_index_pair_v:\n{zero_index_pair_v}')
    print(f'zero_orientation_v:\n{zero_orientation_v}')
    print(f'f_v[zero_index_pair_v]:\n{f_v[zero_index_pair_v]}')
    print(f'\n{np.vstack((zero_index_pair_v.T, zero_orientation_v, t_v[zero_index_pair_v].T, f_v[zero_index_pair_v].T, zero_t_v))}')
    print('\n\n')

    f_v = np.array([0, -2, 0, 2, -2, 2, 2, 0, 0, 0, -2, -2, -3, 0, -3, 0, 0, 3, 0])
    t_v = np.arange(len(f_v))
    zero_index_pair_v, zero_orientation_v, zero_t_v = oriented_zeros(f_v)
    print(f'zero_index_pair_v:\n{zero_index_pair_v}')
    print(f'zero_orientation_v:\n{zero_orientation_v}')
    print(f'f_v[zero_index_pair_v]:\n{f_v[zero_index_pair_v]}')
    print(f'\n{np.vstack((zero_index_pair_v.T, zero_orientation_v, t_v[zero_index_pair_v].T, f_v[zero_index_pair_v].T, zero_t_v))}')
    print('\n\n')

    f_v = np.array([-2, 0, 2, -2, 2, 2, 0, 0, 0, -2, -2, -3, 0, -3, 0, 0, 3])
    t_v = np.arange(len(f_v))
    zero_index_pair_v, zero_orientation_v, zero_t_v = oriented_zeros(f_v, orientation_p=(lambda o:o >= 0))
    print(f'zero_index_pair_v:\n{zero_index_pair_v}')
    print(f'zero_orientation_v:\n{zero_orientation_v}')
    print(f'f_v[zero_index_pair_v]:\n{f_v[zero_index_pair_v]}')
    print(f'\n{np.vstack((zero_index_pair_v.T, zero_orientation_v, t_v[zero_index_pair_v].T, f_v[zero_index_pair_v].T, zero_t_v))}')
    print('\n\n')

    f_v = np.array([0, -2, 0, 2, -2, 2, 2, 0, 0, 0, -2, -2, -3, 0, -3, 0, 0, 3, 0])
    t_v = np.arange(len(f_v))
    zero_index_pair_v, zero_orientation_v, zero_t_v = oriented_zeros(f_v, orientation_p=(lambda o:o == 0))
    print(f'zero_index_pair_v:\n{zero_index_pair_v}')
    print(f'zero_orientation_v:\n{zero_orientation_v}')
    print(f'f_v[zero_index_pair_v]:\n{f_v[zero_index_pair_v]}')
    print(f'\n{np.vstack((zero_index_pair_v.T, zero_orientation_v, t_v[zero_index_pair_v].T, f_v[zero_index_pair_v].T, zero_t_v))}')
    print('\n\n')

    f_v = np.array([0, -2, 0, 2, -2, 2, 2, 0, 0, 0, -2, -2, -3, 0, -3, 0, 0, 3, 0])
    t_v = np.arange(len(f_v))
    zero_index_pair_v, zero_orientation_v, zero_t_v = oriented_zeros(f_v, orientation_p=(lambda o:o != 0))
    print(f'zero_index_pair_v:\n{zero_index_pair_v}')
    print(f'zero_orientation_v:\n{zero_orientation_v}')
    print(f'f_v[zero_index_pair_v]:\n{f_v[zero_index_pair_v]}')
    print(f'\n{np.vstack((zero_index_pair_v.T, zero_orientation_v, t_v[zero_index_pair_v].T, f_v[zero_index_pair_v].T, zero_t_v))}')
    print('\n\n')

    f_v = np.array([0, -2, 0, 2, -2, 2, 2, 0, 0, 0, -2, -2, -3, 0, -3, 0, 0, 3, 0])
    t_v = np.arange(len(f_v))
    zero_index_pair_v, zero_orientation_v, zero_t_v = oriented_zeros(f_v, orientation_p=(lambda o:o <= 0))
    print(f'zero_index_pair_v:\n{zero_index_pair_v}')
    print(f'zero_orientation_v:\n{zero_orientation_v}')
    print(f'f_v[zero_index_pair_v]:\n{f_v[zero_index_pair_v]}')
    print(f'\n{np.vstack((zero_index_pair_v.T, zero_orientation_v, t_v[zero_index_pair_v].T, f_v[zero_index_pair_v].T, zero_t_v))}')
    print('\n\n')

    f_v = np.array([0, -2, 0, 2, -2, 2, 2, 0, 0, 0, -2, -2, -3, 0, -3, 0, 0, 3, 0])
    t_v = np.arange(len(f_v))
    zero_index_pair_v, zero_orientation_v, zero_t_v = oriented_zeros(f_v, orientation_p=(lambda o:o < 0))
    print(f'zero_index_pair_v:\n{zero_index_pair_v}')
    print(f'zero_orientation_v:\n{zero_orientation_v}')
    print(f'f_v[zero_index_pair_v]:\n{f_v[zero_index_pair_v]}')
    print(f'\n{np.vstack((zero_index_pair_v.T, zero_orientation_v, t_v[zero_index_pair_v].T, f_v[zero_index_pair_v].T, zero_t_v))}')
    print('\n\n')

    f_v = np.array([0, -2, 0, 2, -2, 2, 2, 0, 0, 0, -2, -2, -3, 0, -3, 0, 0, 3, 0])
    t_v = np.arange(len(f_v))
    zero_index_pair_v, zero_orientation_v, zero_t_v = oriented_zeros(f_v, orientation_p=(lambda o:False))
    print(f'zero_index_pair_v:\n{zero_index_pair_v}')
    print(f'zero_index_pair_v.shape:\n{zero_index_pair_v.shape}')
    print(f'zero_orientation_v:\n{zero_orientation_v}')
    print(f'f_v[zero_index_pair_v]:\n{f_v[zero_index_pair_v]}')
    print(f'\n{np.vstack((zero_index_pair_v.T, zero_orientation_v, t_v[zero_index_pair_v].T, f_v[zero_index_pair_v].T, zero_t_v))}')
    print('\n\n')

    f_v = np.array([0, -2, 0, 2, -2, 2, 2, 0, 0, 0, -2, -2, -3, 0, -3, 0, 0, 3, 0])
    t_v = np.linspace(0.0, 10.0, len(f_v))
    zero_index_pair_v, zero_orientation_v, zero_t_v = oriented_zeros(f_v, t_v=t_v)
    print(f'zero_index_pair_v:\n{zero_index_pair_v}')
    print(f'zero_index_pair_v.shape:\n{zero_index_pair_v.shape}')
    print(f'zero_orientation_v:\n{zero_orientation_v}')
    print(f'f_v[zero_index_pair_v]:\n{f_v[zero_index_pair_v]}')
    print(f'\n{np.vstack((zero_index_pair_v.T, zero_orientation_v, t_v[zero_index_pair_v].T, f_v[zero_index_pair_v].T, zero_t_v))}')
    print('\n\n')

    cp_index_pair_v, cp_orientation_v, cp_t_v = critical_points(f_v, t_v=t_v)
    print(f'cp_index_pair_v:\n{cp_index_pair_v}')
    print(f'cp_orientation_v:\n{cp_orientation_v}')
    print(f'cp_t_v:\n{cp_t_v}')
    print('\n\n')

