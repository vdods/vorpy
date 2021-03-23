"""
Module for working with Bezier curves/patches.
"""

import numpy as np
import sympy as sp
import typing
import vorpy.realfunction.bernstein
import vorpy.realfunction.bezier
import vorpy.symbolic

def _numerical_derivative_two_sided (func:typing.Callable[[float],float], x_0:float) -> typing.Tuple[float,float]:
    """This is a rather primitive way to compute the derivative, but hey."""
    epsilon = 1.0e-9
    func_x_0 = func(x_0)
    deriv_neg = (func_x_0 - func(x_0-epsilon)) / epsilon
    deriv_pos = (func(x_0+epsilon) - func_x_0) / epsilon
    return deriv_neg, deriv_pos

def _numerical_derivative (func:typing.Callable[[float],float], x_0:float) -> float:
    deriv_neg, deriv_pos = _numerical_derivative_two_sided(func, x_0)
    if np.max(np.abs(deriv_neg - deriv_pos)) > 1.0e-6: # Sort of arbitrary, doesn't take into account magnitudes.
        raise ValueError(f'func is not contiuously differentiable; deriv_neg = {deriv_neg}, deriv_pos = {deriv_pos}; x_0 = {x_0}')
    return 0.5*(deriv_neg+deriv_pos)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pathlib
    import scipy.interpolate
    import sys

    cubic_bezier_t = vorpy.realfunction.bezier.bezier_tensor(degree=3, jet=1)
    # Express the Bezier tensor in polynomial form.
    t = sp.var('t')
    print(vorpy.tensor.contract('k,kbj', vorpy.realfunction.bernstein.bernstein_polynomial_basis(t, degree=3), cubic_bezier_t, dtype=object))

    y_jet_t = np.array([
        [0.0,  1.0],
        [0.5,  0.25],
        [0.75, 0.0],
        [0.25, -1.0],
    ])
    t_v = np.array([0.0, 1.0, 1.1, 1.375])

    linear_interpolator = scipy.interpolate.interp1d(t_v, y_jet_t[:,0])
    cubic_interpolator = vorpy.realfunction.bezier.cubic_interpolation(t_v, y_jet_t)

    deriv_v = np.array([_numerical_derivative(cubic_interpolator, t) for t in t_v])
    print(f'np.abs(deriv_v - y_jet_t[:,1]) = {np.abs(deriv_v - y_jet_t[:,1])}')
    assert np.all(np.abs(deriv_v - y_jet_t[:,1]) < 1.0e-7)

    if False:
        row_count   = 1
        col_count   = 1
        size        = 20
        fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

        plot_t_linear_v = np.linspace(t_v[0], t_v[-1], 2001)
        plot_t_cubic_v  = np.linspace(t_v[0]-0.2, t_v[-1]+0.2, 2001)

        axis = axis_vv[0][0]
        axis.plot(plot_t_linear_v, linear_interpolator(plot_t_linear_v))
        axis.plot(plot_t_cubic_v, cubic_interpolator(plot_t_cubic_v))

        plot_p = pathlib.Path(f'cubic-bezier.png')

        fig.tight_layout()
        plot_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(plot_p), bbox_inches='tight')
        print(f'wrote to file "{plot_p}"')
        # VERY important to do this -- otherwise your memory will slowly fill up!
        # Not sure which one is actually sufficient -- apparently none of them are, YAY!
        plt.clf()
        plt.cla()
        plt.close()
        plt.close(fig)
        plt.close('all')
        del fig
        del axis_vv

    if True:
        np.random.seed(42)

        # Test tensor-valued cubic Bezier interpolation
        t_v = np.cumsum(np.abs(np.random.randn(10)))
        y_jet_t = np.random.randn(len(t_v),2,2,1)

        cubic_interpolator = vorpy.realfunction.bezier.cubic_interpolation(t_v, y_jet_t)
        interp_t_v = np.linspace(t_v[0], t_v[-1], 1000)
        interp_y_t = cubic_interpolator(interp_t_v)

        deriv_t = np.ndarray((len(t_v),)+(2,1), dtype=float)
        for i,t in enumerate(t_v):
            deriv_t[i,...] = _numerical_derivative(cubic_interpolator, t)
        print(f'deriv_t.shape = {deriv_t.shape}')
        print(f'np.max(np.abs(deriv_t - y_jet_t[:,1])) = {np.max(np.abs(deriv_t - y_jet_t[:,1]))}')
        assert np.all(np.abs(deriv_t - y_jet_t[:,1,...]) < 3.0e-7)

    sys.exit(0)

    # Solve Hamilton's equations for some system and use a cubic Bezier to interpolate the solution
    # between the numerically integrated solution points.

    import pathlib
    import vorpy.experimental.kh

    x_initial = 1.0
    p_y_initial = 0.2
    H_initial = 0.0
    solution_sheet = 0
    qp_initial = vorpy.experimental.kh.EuclideanNumerics.qp_constrained_by_H__fast(
        np.array([x_initial, 0.0, 0.0, 0.0, p_y_initial, H_initial])
    )[solution_sheet]
    results = vorpy.experimental.kh.EuclideanNumerics.compute_trajectory(
        pickle_filename_p=pathlib.Path('kh_solution_bezier.pickle'),
        qp_initial=qp_initial,
        t_final=40.0,
        solution_sheet=solution_sheet,
        return_y_jet=True,
    )

    ## Compute the 1-jet of the qp trajectory by using the fact that d(qp)/dt is given by
    ## the symplectic gradient of H.
    #jet_qp_t = np.ndarray((2,)+results.y_t.shape, dtype=float)
    #jet_qp_t[0,...] = results.y_t.shape
    #jet_qp_t[1,...] = vorpy.apply_along_axes(vorpy.experimental.kh.EuclideanNumerics.X_H__fast, (1,2), (results.y_t,))

    # TODO: Figure out how to compensate for the fact that the solution is non-uniformly sampled in time.
    # TODO: Use the fact that the p part of qp is related to dq/dt via the Legendre transform, and that a
    # quartic Bezier curve (maybe represented as a quintic using the jet boundaries?) can be computed for q.


