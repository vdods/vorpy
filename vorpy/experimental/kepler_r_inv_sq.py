import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sympy as sp
import typing
import vorpy.experimental.plot
import vorpy.integration.adaptive
import vorpy.linalg
import vorpy.manifold
import vorpy.symbolic
import vorpy.symplectic
import vorpy.tensor

def phase_space_coordinates () -> np.ndarray:
    x  ,y   = q = np.array(sp.var('x,y'))
    p_x,p_y = p = np.array(sp.var('p_x,p_y'))
    qp = np.array([q,p])
    return qp

def dilation_by (lam:typing.Any) -> typing.Callable[[np.ndarray], np.ndarray]:
    def dilation (qp:np.ndarray) -> np.ndarray:
        q, p = qp
        return np.array([q*lam, p/lam])
    return dilation

def K (qp:np.ndarray) -> typing.Any:
    q, p = qp
    return np.dot(p,p)/2

def V (qp:np.ndarray) -> typing.Any:
    q, p = qp
    return -1/np.dot(q,q)

def H (qp:np.ndarray) -> typing.Any:
    return K(qp) + V(qp)

def X_H (qp:np.ndarray) -> np.ndarray:
    return vorpy.symplectic.symplectic_gradient_of(H(qp), qp)

def p_theta (qp:np.ndarray) -> typing.Any:
    X = vorpy.linalg.scalar_cross_product_tensor(dtype=sp.Integer)
    q, p = qp
    return vorpy.tensor.contract('jk,j,k', X, q, p, dtype=object)

def J (qp:np.ndarray) -> np.ndarray:
    q, p = qp
    return np.dot(q, p)

@vorpy.symbolic.cache_lambdify(
    function_id='H',
    argument_id='qp',
    replacement_d={'dtype=object':'dtype=float', 'ndarray':'np.ndarray'},
    import_v=['import numpy as np'],
    verbose=True,
)
def H__fast ():
    qp = phase_space_coordinates()
    return H(qp), qp

@vorpy.symbolic.cache_lambdify(
    function_id='X_H',
    argument_id='qp',
    replacement_d={'dtype=object':'dtype=float', 'ndarray':'np.ndarray'},
    import_v=['import numpy as np'],
    verbose=True,
)
def X_H__fast ():
    qp = phase_space_coordinates()
    return X_H(qp), qp

@vorpy.symbolic.cache_lambdify(
    function_id='p_theta',
    argument_id='qp',
    replacement_d={'dtype=object':'dtype=float', 'ndarray':'np.ndarray'},
    import_v=['import numpy as np'],
    verbose=True,
)
def p_theta__fast ():
    qp = phase_space_coordinates()
    return p_theta(qp), qp

def compute_trajectory (qp_initial:np.ndarray, t_final:float) -> vorpy.integration.adaptive.IntegrateVectorFieldResults:
    H_initial = H__fast(qp_initial)
    p_theta_initial = p_theta__fast(qp_initial)

    H_cq = vorpy.integration.adaptive.ControlledQuantity(
        name='H',
        reference_quantity=H_initial,
        global_error_band=vorpy.integration.adaptive.RealInterval(1.0e-8, 1.0e-6),
        quantity_evaluator=(lambda t,qp:typing.cast(float, H__fast(qp))), # type: ignore
    )
    p_theta_cq = vorpy.integration.adaptive.ControlledQuantity(
        name='p_theta',
        reference_quantity=p_theta_initial,
        global_error_band=vorpy.integration.adaptive.RealInterval(1.0e-8, 1.0e-6),
        quantity_evaluator=(lambda t,qp:typing.cast(float, p_theta__fast(qp))), # type: ignore
    )
    controlled_sq_ltee = vorpy.integration.adaptive.ControlledSquaredLTEE(
        global_error_band=vorpy.integration.adaptive.RealInterval(1.0e-8**2, 1.0e-6**2),
    )

    results = vorpy.integration.adaptive.integrate_vector_field(
        vector_field=(lambda t,qp:X_H__fast(qp)), # type: ignore
        t_initial=0.0,
        y_initial=qp_initial,
        t_final=t_final,
        controlled_quantity_d={
            'H abs error':H_cq,
            'p_theta abs error':p_theta_cq,
        },
        controlled_sq_ltee=controlled_sq_ltee,
        return_y_jet=False,
    )
    return results

def do_stuff ():
    x  ,y   = q = np.array(sp.var('x,y'))
    p_x,p_y = p = np.array(sp.var('p_x,p_y'))

    qp = np.array([q,p])

    lam = sp.var('lam')
    dilation = dilation_by(lam)

    # Prove that kinetic, potential, total energy functions are homogeneous of degree -2 under dilation action.
    assert np.all(sp.simplify(K(dilation(qp)) - K(qp)/lam**2) == 0)
    assert np.all(sp.simplify(V(dilation(qp)) - V(qp)/lam**2) == 0)
    assert np.all(sp.simplify(H(dilation(qp)) - H(qp)/lam**2) == 0)

    print(f'X_H:\n{X_H(qp)}')

    # Just want to know that our explicit formula is correct.
    assert np.all(X_H(qp) - np.array([p, -2*q/np.dot(q,q)**2]) == 0)

    # Prove that p_theta is an integral of motion (it is invariant along X_H)
    assert vorpy.manifold.directional_derivative(X_H(qp), p_theta(qp), qp, post_process_o=sp.simplify) == 0

    # J doesn't appear to be any kind of integral
    print('X_H(J):')
    print(vorpy.manifold.directional_derivative(X_H(qp), J(qp), qp, post_process_o=sp.simplify))

    epsilon = 0.03
    qp_initial = np.array([
        [1.0, 0.0],
        [0.0, np.sqrt(2.0)+epsilon],
    ])
    t_final = 3000.0
    results = compute_trajectory(qp_initial, t_final)

    plot = vorpy.experimental.plot.Plot(row_count=1, col_count=2, size=10)

    axis = plot.axis(0, 0)
    axis.set_title('(x(t), y(t))')
    axis.set_aspect(1.0)
    axis.plot([0.0], [0.0], '.', color='black')
    axis.plot(results.y_t[:,0,0], results.y_t[:,0,1])

    axis = plot.axis(0, 1)
    axis.set_title('(t, R(t)) -- R := r^2')
    x_v = results.y_t[:,0,0]
    y_v = results.y_t[:,0,1]
    R_v = x_v**2 + y_v**2
    axis.semilogy(results.t_v, R_v)

    plot.savefig(pathlib.Path('kepler_r_inv_sq/dilating_outward.png'))

if __name__ == '__main__':
    do_stuff()
