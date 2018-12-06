import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sympy as sp
import typing
import vorpy.experimental.integrate_tangent_flow
import vorpy.integration
import vorpy.integration.adaptive
import vorpy.symbolic
import vorpy.symplectic

def symplectic_gradient_of (f:typing.Any, qp:np.ndarray, omega_inv:np.ndarray) -> np.ndarray:
    """
    Returns the symplectic gradient of the function f with respect to Darboux coordinates qp.
    In particular, this is defined to be the symplectic dual of the covector field df.

    The function f may be a tensor quantity, in which case the return value has shape

        np.shape(f) + np.shape(qp)

    and in particular, if f is a scalar function, the return value has shape equal to that of qp
    (this is the standard case).

    There is a choice of convention regarding where a particular negative sign goes (which
    stems from the choice of sign in the definition of the canonical symplectic form).  See
    the documentation for canonical_symplectic_form_abstract for more on this.  The short story is that
    the convention is picked such that the symplectic gradient of function H(qp) gives the
    vector field whose flow equation is Hamilton's equations where H is the Hamiltonian function.
    """
    vorpy.symplectic.validate_darboux_coordinates_quantity_or_raise(qp, quantity_name='qp')
    df = vorpy.symbolic.differential(f, qp)
    assert df.shape == np.shape(f) + qp.shape
    return vorpy.tensor.contract('ijkl,kl', omega_inv, df, dtype=object)

def do_stuff ():
    # Dimension of configuration space
    d = 2
    # Position coordinates
    x, y = q = np.array(sp.var('x,y'))
    # Momentum coordinates
    p_x, p_y = p = np.array(sp.var('p_x,p_y'))

    # Charge
    #e = sp.var('e') # TODO: Could do symbolic value later
    e = sp.Integer(-1)
    # Mass
    #m = sp.var('m') # TODO: Could do symbolic value later
    m = sp.Integer(1)

    # Has shape (2,3), where 2 is the "phase axis", i.e. position=0 or momentum=1.
    # So qp[0] is q, and qp[1] is p.
    qp = np.array([q, p])
    qp_shape = qp.shape

    # Let's make a symbolic identity 2x2 matrix
    I = np.eye(d, dtype=sp.Integer)

    r_squared = x**2 + y**2
    r = sp.sqrt(r_squared)
    b = 1 / (1-r)

    B = np.array([
        [ 0, b],
        [-b, 0],
    ])
    omega = np.ndarray(qp_shape+qp_shape, dtype=object)
    omega[0,:,0,:] = B
    omega[0,:,1,:] = -I
    omega[1,:,0,:] = I
    omega[1,:,1,:] = 0

    omega_inv = np.ndarray(qp_shape+qp_shape, dtype=object)
    omega_inv[0,:,0,:] = 0
    omega_inv[0,:,1,:] = I
    omega_inv[1,:,0,:] = -I
    omega_inv[1,:,1,:] = B

    # Verify that omega_inv is correct
    omega_as_matrix = omega.reshape(2*d, 2*d)
    omega_inv_as_matrix = omega_inv.reshape(2*d, 2*d)
    assert np.all(np.dot(omega_as_matrix, omega_inv_as_matrix) == np.eye(2*d, dtype=sp.Integer))
    assert np.all(np.dot(omega_inv_as_matrix, omega_as_matrix) == np.eye(2*d, dtype=sp.Integer))
    print('omega_inv check passed')

    # Write down the dynamics

    # Kinetic energy
    K = np.dot(p, p)/(2*m)
    # Total energy
    H = K

    # Hamiltonian vector field
    X = symplectic_gradient_of(H, qp, omega_inv)
    # Also compute the Jacobian of X, so that the tangent map of the flow, along the flow curve, can be computed.
    DX = vorpy.symbolic.differential(X, qp)

    J = vorpy.symbolic.tensor('J', qp.shape+qp.shape)
    S_cond = vorpy.symplectic.symplectomorphicity_condition(J, dtype=sp.Integer, return_as_scalar_if_possible=True)
    S = sp.sqrt(np.sum(np.square(S_cond))).simplify()

    print(f'H = {H}')
    print(f'X = {X}')
    print(f'DX = {DX}')

    replacement_d = {
        'array':'np.array',
        'sqrt':'np.sqrt',
        'dtype=object':'dtype=float',
    }
    # Compute "fast" python functions for everything
    X_fast = vorpy.symbolic.lambdified(X, qp, replacement_d=replacement_d, verbose=True)
    DX_fast = vorpy.symbolic.lambdified(DX, qp, replacement_d=replacement_d, verbose=True)
    H_fast = vorpy.symbolic.lambdified(H, qp, replacement_d=replacement_d, verbose=True)
    S_cond_fast = vorpy.symbolic.lambdified(S_cond, J, replacement_d=replacement_d, verbose=True)
    S_fast = vorpy.symbolic.lambdified(S, J, replacement_d=replacement_d, verbose=True)

    # Here is where we choose initial conditions
    qp_initial = np.array([
        [0.5, 0.0],
        [1.0, 1.0],
    ])
    t_initial = 0.0
    t_final = 200.0

    # Now to actually run the numerical integration.
    y_shape = qp.shape
    J_shape = qp.shape+qp.shape

    #print(f'processing y_initial = {y_initial}')
    H_initial = H_fast(qp_initial)
    S_initial = 0.0 # By definition, df at t_initial is the identity matrix, and is therefore trivially symplectic.
    controlled_quantity_v = [
        vorpy.integration.adaptive.ControlledQuantity(
            name='H',
            reference_quantity=H_initial,
            quantity_evaluator=(lambda t,z:H_fast(z[:qp_initial.size].reshape(qp.shape))),
            global_error_band=vorpy.integration.adaptive.RealInterval(10e-10, 10e-7),
            #global_error_band=vorpy.integration.adaptive.RealInterval(10e-10, 10e-7),
        ),
        #vorpy.integration.adaptive.ControlledQuantity(
            #name='S',
            #reference_quantity=S_initial,
            #quantity_evaluator=(lambda t,z:S_fast(z[qp_initial.size:].reshape(J_shape))),
            #global_error_band=vorpy.integration.adaptive.RealInterval(10e-10, 10e8),
            ##global_error_band=vorpy.integration.adaptive.RealInterval(10e-30, 10e-6),
            ##global_error_band=vorpy.integration.adaptive.RealInterval(10e-8, 10e-6),
        #),
    ]
    results = vorpy.experimental.integrate_tangent_flow.integrate_tangent_map(
        x=X_fast,
        Dx=DX_fast,
        t_initial=t_initial,
        t_final=t_final,
        y_initial=qp_initial,
        controlled_quantity_d={cq.name():cq for cq in controlled_quantity_v},
        controlled_sq_ltee=vorpy.integration.adaptive.ControlledSquaredLTEE(global_error_band=vorpy.integration.adaptive.RealInterval(1.0e-9**2, 1.0e-7**2)),
    )

    print(f'final time integrated to = {results.t_v[-1]}')

    # Plot everything
    row_count   = 1
    col_count   = 2
    size        = 8
    fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

    axis = axis_vv[0][0]
    axis.set_title('(x(t), y(t))')
    axis.set_aspect(1.0)
    axis.plot(results.y_t[:,0,0], results.y_t[:,0,1])

    axis = axis_vv[0][1]
    axis.set_title('(p_x(t), p_y(t))')
    axis.set_aspect(1.0)
    axis.plot(results.y_t[:,1,0], results.y_t[:,1,1])

    plot_p = pathlib.Path('magnetic.png')
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


if __name__ == '__main__':
    do_stuff()

