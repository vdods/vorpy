"""
Design notes for an integrator that numerically approximates the tangent map of the flow of a vector
field along a flow curve.  Let I denote a real interval, used for parameterizing curves.

This is conceptually equivalent to solving for the vector field induced along a flow curve due to
variations in its initial conditions.

The flow F : I x M -> M of vector field X : Gamma(TM) is defined by the ODE

    d/dt F(t,m) = X(F(t,m)) for all (t,m) : I x M.

It can be shown that this induces an ODE on its tangent map, TF : TM -> TM, which is

    d/dt (dF/dM)*v = TX*(dF/dM)*v,

where v : TM and dF/dM : I x TM -> TM is the partial derivative of F with respect to the M component.  In
coordinates, this breaks down into two ODEs, one being the original flow ODE, and the other being an ODE
on the coordinate-specific matrix expression for the action of dF/dM on the fiber of TM.

In coordinates, Let V be the fiber of TM, and let D indicate the elementary differential operator on the
coordinate vector space.  Let f denote the coordinate expression for the flow map F.  Then

    X(m)             = (m,x(m)) for some map x : M -> V,
    TX(m,v)          = ((m,v), (x(m),Dx(m)*v)) for v : V, noting that Dx : M -> V \otimes V^*,
                     = ((m,x(m)), (v,Dx(m)*v)) written with inner fiber components transposed,
    dF/dM ((t,m), v) = (f(t,m), Df(t,m)*v).

The ODE is written in coordinates as

    d/dt (f(t,m), Df(t,m)*v) = TX(dF/dM ((t,m),v))
                             = TX(f(t,m), Df(t,m)*v)
                             = ((f(t,m),x(f(t,m))), (Df(t,m)*v,Dx(f(t,m))*Df(t,m)*v)).

The basepoint for each can be dropped since coordinates are being used, giving

    d/dt (f(t,m), Df(t,m)*v) = (x(f(t,m)), Dx(f(t,m))*Df(t,m)*v),

which is actually the two separate ODEs

    d/dt f(t,m)    = x(f(t,m)),
    d/dt Df(t,m)*v = Dx(f(t,m))*Df(t,m)*v.

Because the natural pairing with v commutes with the time derivative, and v is arbitrary in V, it follows that

    d/dt f(t,m)  = x(f(t,m)),
    d/dt Df(t,m) = Dx(f(t,m))*Df(t,m),

where the the ODEs are in a vector value f(t,m) and matrix value Df(t,m) respectively.

To phrase this as an initial value problem, let y(t) denote f(t,m) for some fixed m : M, and let J(t) denote
Df(t,m), noting that y(t) is vector valued, and J(t) is matrix-valued.  Because J(0) = Df(0,m), and the flow
at time zero is defined to be the identity map, it follows that J(0) is the identity matrix.  The ODEs are

    d/dt y(t) = x(y(t)),
    d/dt J(t) = Dx(y(t))*J(t).

This can be written as the flow of the vector field W on (M, V \otimes V^*).

    W(y, J) = (x(y), Dx(y)*J).

Note that the expression Dx(y)*J is composition of matrices (not a full contraction), resulting in a matrix,

If X is the Hamiltonian vector field for some Hamiltonian function H, then its flow F is a family of
symplectomorphisms, and therefore each respect the symplectic form.  Thus J must satisfy a pullback
identity involving the symplectic form.

TODO: Write down the non-autonomous equations (i.e. where X depends on t also).
"""

import itertools
import numpy as np
import sys
import typing
import vorpy.integration.adaptive
import vorpy.pickle

class IntegrateTangentMapResults:
    def __init__ (
        self,
        *,
        t_v:np.ndarray,
        y_t:np.ndarray,
        J_t:np.ndarray,
        global_error_vd:np.ndarray,
        local_error_vd:np.ndarray,
        t_step_v:np.ndarray,
        t_step_iteration_count_v:np.ndarray,
        failure_explanation_o:typing.Optional[str],
    ) -> None:
        # Check the identity claimed for t_v and t_step_v.
        identity_failure_v = (t_v[:-2]+t_step_v[:-1]) - t_v[1:-1]
        if len(identity_failure_v) > 0:
            #print(f'max identity failure: {np.max(np.abs(identity_failure_v))}')
            assert np.max(np.abs(identity_failure_v)) == 0
            #print(f'max naive identity failure: {np.max(np.abs(np.diff(t_v[:-1]) - t_step_v[:-1]))}')

        # Sequence of time values, indexed as t_v[i].
        self.t_v                    = t_v
        # Sequence (tensor) of parameter values, indexed as y_t[i,J], where i is the time index and J
        # is the [multi]index for the parameter type (could be scalar, vector, or tensor).
        self.y_t                    = y_t
        # Sequence (tensor) of parameter values, indexed as J_t[i,J,K], where i is the time index and J and K
        # are the [multi]indices for the parameter type (could be scalar, vector, or tensor).
        self.J_t                    = J_t
        # Dictionary of global error sequences mapped to their names.  Each global error sequence is indexed
        # as global_error_v[i], where i is the index for t_v.
        self.global_error_vd        = global_error_vd
        # Dictionary of local error sequences mapped to their names.  Each local error sequence is indexed
        # as local_error_v[i], where i is the index for t_v.
        self.local_error_vd         = local_error_vd
        # Sequence of timestep values, indexed as t_step_v[i], though len(t_step_v) == len(t_v)-1.  Note that
        # this should satisfy t_v[:-1]+t_step_v == t_v[1:] (since each time value is defined as the previous
        # time value plus the current time step), but it will NOT satisfy t_v[1:]-t_v[:-1] == t_step_v due to
        # numerical roundoff error.
        self.t_step_v               = t_step_v
        # Number of iterations it took to compute an acceptable t_step value.  Indexed as t_step_iteration_count_v[i],
        # where i is the index for t_v.
        self.t_step_iteration_count_v   = t_step_iteration_count_v
        # If failure_explanation_o is None, then the integration is understood to have succeeded.
        self.succeeded              = failure_explanation_o is None
        # Store the [optional] failure explanation.
        self.failure_explanation_o  = failure_explanation_o

def integrate_tangent_map (
    *,
    x:typing.Callable[[np.ndarray],np.ndarray],
    Dx:typing.Callable[[np.ndarray],np.ndarray],
    t_initial:float,
    y_initial:np.ndarray,
    t_final:float,
    controlled_quantity_d:typing.Dict[str,vorpy.integration.adaptive.ControlledQuantity],
    controlled_sq_ltee:vorpy.integration.adaptive.ControlledSquaredLTEE,
) -> IntegrateTangentMapResults:
    """
    The vector field will be over manifold having coordinates (y,J), where y has shape s_y and
    J has shape s_y+s_y (i.e. is a matrix operating on y).
    """

    y_shape = y_initial.shape
    J_shape = y_shape + y_shape
    base_space_dim = y_initial.size
    #J_initial = np.eye(base_space_dim, dtype=float).reshape(J_shape)
    J_initial = vorpy.tensor.identity_tensor(y_shape, dtype=float)
    assert J_initial.shape == J_shape
    z_shape = (y_initial.size + J_initial.size,)

    # TODO: Make this a more formal part of an API for integrate_tangent_map
    def y_view (z:np.ndarray) -> np.ndarray:
        return z[:base_space_dim].reshape(y_shape)

    # TODO: Make this a more formal part of an API for integrate_tangent_map
    def J_view (z:np.ndarray) -> np.ndarray:
        return z[base_space_dim:].reshape(J_shape)

    # TODO: Make this a more formal part of an API for integrate_tangent_map
    def z_from (y:np.ndarray, J:np.ndarray) -> np.ndarray:
        z = np.ndarray(z_shape, dtype=float)
        y_view(z)[...] = y
        J_view(z)[...] = J
        return z

    print(f'y_initial = {y_initial}')
    print(f'J_initial = {J_initial}')
    z_initial = z_from(y_initial, J_initial)
    print(f'z_initial = {z_initial}')

    def vector_field (t:float, z:np.ndarray) -> np.ndarray:
        y = y_view(z)
        J = J_view(z)
        # Dx produces shape J_shape, and must operate on J (which has shape s_y+s_y), but in order to do this
        # in a reasonable way, we'll flatten it, so it just has shape (base_space_dim,base_space_dim).
        Dx_flat = Dx(y).reshape(base_space_dim, base_space_dim)
        J_flat = J.reshape(base_space_dim, base_space_dim)
        return z_from(x(y), np.dot(Dx_flat, J_flat).reshape(J_shape))

    results = vorpy.integration.adaptive.integrate_vector_field(
        vector_field=vector_field,
        t_initial=t_initial,
        y_initial=z_initial,
        t_final=t_final,
        controlled_quantity_d=controlled_quantity_d,
        controlled_sq_ltee=controlled_sq_ltee,
    )

    return IntegrateTangentMapResults(
        t_v=results.t_v,
        y_t=results.y_t[:,:base_space_dim].reshape(-1,*y_shape),
        J_t=results.y_t[:,base_space_dim:].reshape(-1,*J_shape),
        global_error_vd=results.global_error_vd,
        local_error_vd=results.local_error_vd,
        t_step_v=results.t_step_v,
        t_step_iteration_count_v=results.t_step_iteration_count_v,
        failure_explanation_o=results.failure_explanation_o,
    )

if __name__ == '__main__':
    # Simple dynamical system -- pendulum

    import matplotlib.pyplot as plt
    import numpy.linalg
    import pathlib
    import sympy as sp
    import vorpy.symplectic

    def svd (M:np.ndarray) -> np.ndarray:
        operand_shape = vorpy.tensor.operand_shape_of(M)
        operand_space_dim = vorpy.tensor.dimension_of_shape(operand_shape)
        M_as_2_tensor = M.reshape(operand_space_dim, operand_space_dim)
        return numpy.linalg.svd(M_as_2_tensor, full_matrices=False, compute_uv=False)

    def plot_dynamics (plot_p, t_initial, t_final, y_initial_v, X_fast, DX_fast, H_fast, S_fast, apply_along_y_t_axes, apply_along_J_t_axes, *, plot_function_o=None, plot_function_2_o=None, write_pickle:bool=False):
        row_count   = 2
        col_count   = 5
        size        = 8
        fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

        for y_initial in y_initial_v:
            y_shape = y_initial.shape
            J_shape = y_shape+y_shape
            print(f'processing y_initial = {y_initial}')
            H_initial = H_fast(y_initial)
            S_initial = 0.0 # By definition, df at t_initial is the identity matrix, and is therefore trivially symplectic.
            controlled_quantity_v = [
                vorpy.integration.adaptive.ControlledQuantity(
                    name='H',
                    reference_quantity=H_initial,
                    quantity_evaluator=(lambda t,z:H_fast(z[:y_initial.size].reshape(y_shape))),
                    global_error_band=vorpy.integration.adaptive.RealInterval(10e-30, 10e-7),
                    #global_error_band=vorpy.integration.adaptive.RealInterval(10e-10, 10e-7),
                ),
                vorpy.integration.adaptive.ControlledQuantity(
                    name='S',
                    reference_quantity=S_initial,
                    quantity_evaluator=(lambda t,z:S_fast(z[y_initial.size:].reshape(J_shape))),
                    global_error_band=vorpy.integration.adaptive.RealInterval(10e-30, 10e3),
                    #global_error_band=vorpy.integration.adaptive.RealInterval(10e-30, 10e-6),
                    #global_error_band=vorpy.integration.adaptive.RealInterval(10e-8, 10e-6),
                ),
            ]
            results = integrate_tangent_map(
                x=X_fast,
                Dx=DX_fast,
                t_initial=t_initial,
                t_final=t_final,
                y_initial=y_initial,
                controlled_quantity_d={cq.name():cq for cq in controlled_quantity_v},
                controlled_sq_ltee=vorpy.integration.adaptive.ControlledSquaredLTEE(global_error_band=vorpy.integration.adaptive.RealInterval(1.0e-9**2, 1.0e-7**2)),
            )

            print(f'len(results.t_v) = {len(results.t_v)}')
            print(f'results.global_error_vd.keys() = {results.global_error_vd.keys()}')
            print(f'results.J_t[-1] = {results.J_t[-1]}')

            H_v = vorpy.apply_along_axes(H_fast, apply_along_y_t_axes, (results.y_t,))
            S_v = vorpy.apply_along_axes(S_fast, apply_along_J_t_axes, (results.J_t,))

            svd_t = vorpy.apply_along_axes(svd, apply_along_J_t_axes, (results.J_t,))
            assert svd_t.shape == (len(results.t_v), y_initial.size), f'expected svd_t.shape == {(len(results.t_v), y_initial.size)} but it was actually {svd_t.shape}'
            svd_lyapunov_exponent_t = np.log(svd_t) / results.t_v[:,np.newaxis]

            print(f'final Lyapunov exponents: {svd_lyapunov_exponent_t[-1]}')

            def condition_number (J:np.ndarray):
                return np.linalg.cond(vorpy.tensor.as_linear_operator(J))

            J_condition_number_v = vorpy.apply_along_axes(condition_number, apply_along_J_t_axes, (results.J_t,))

            if write_pickle:
                data_d = dict(
                    results=results,
                    y_initial=y_initial,
                    y_shape=y_shape,
                    J_shape=J_shape,
                    H_initial=H_initial,
                    S_initial=S_initial,
                    H_v=H_v,
                    S_v=S_v,
                    svd_t=svd_t,
                    svd_lyapunov_exponent_t=svd_lyapunov_exponent_t,
                    J_condition_number_v=J_condition_number_v,
                )
                pickle_p = pathlib.Path(f'{plot_p}.pickle')
                pickle_p.parent.mkdir(parents=True, exist_ok=True)
                vorpy.pickle.pickle(data=data_d, pickle_filename=pickle_p, log_out=sys.stdout)

            ## In theory this should give the same result as the SVD-based computation, but there are
            ## more operations here (taking the symmetric square of J_t).
            #symmetrized_A_t = np.einsum('ijk,ijl->ikl', results.J_t, results.J_t)
            #eigenvalues_t = vorpy.apply_along_axes(numpy.linalg.eigvalsh, apply_along_J_t_axes, (symmetrized_A_t,))
            #assert eigenvalues_t.shape == (len(results.t_v), 2)
            ## numpy.linalg.eigvalsh gives eigenvalues in ascending order, but we want descending.
            #eigenvalues_t = eigenvalues_t[:,::-1] # Slice that reverses the eigenvalue order
            #eigenvalues_lyapunov_exponent_v = np.log(eigenvalues_t) / (2*results.t_v[:,np.newaxis])

            axis = axis_vv[0][0]
            if plot_function_o is not None:
                plot_function_o(axis, results)
            elif y_initial.size == 2:
                axis.set_title('phase space')
                axis.set_aspect(1.0)
                axis.plot(results.y_t[:,0], results.y_t[:,1])
            elif vorpy.symplectic.is_darboux_coordinates_quantity(y_initial) and y_initial[0].size == 2:
                # Technically you can't determine that it's a Darboux quantity from just the shape, but it's
                # a pretty reasonable assumption here, since we're only doing Hamiltonian mechanics.
                # Plot positions only.
                q_t = results.y_t[:,0,...].reshape(-1,2)
                axis.set_title('positions as functions of t')
                axis.plot(q_t[:,0], q_t[:,1])
            else:
                axis.set_title('coordinates as functions of t')
                axis.plot(results.t_v, results.y_t.reshape(-1, y_initial.size))

            axis = axis_vv[1][0]
            if plot_function_2_o is not None:
                plot_function_2_o(axis, results)

            axis = axis_vv[0][1]
            axis.set_title('time steps')
            axis.semilogy(results.t_v[:-1], results.t_step_v, '.', alpha=0.2)

            axis = axis_vv[1][1]
            sq_ltee = results.global_error_vd['Squared LTEE']
            axis.set_title(f'LTEE squared - max: {np.max(sq_ltee)}')
            axis.semilogy(results.t_v, sq_ltee)

            axis = axis_vv[0][2]
            #abs_H_error = np.abs(H_v - H_v[0])
            axis.set_title(f'abs(H - H_0) - max: {np.max(results.global_error_vd["H"]):.3e}\nglobal:blue, local:green')
            #axis.semilogy(results.t_v, abs_H_error)
            axis.semilogy(results.t_v, results.global_error_vd['H'], '.', color='blue')
            axis.semilogy(results.t_v, results.local_error_vd['H'], '.', color='green')

            axis = axis_vv[1][2]
            #axis.set_title(f'symplectomorphicity_condition - max: {np.max(S_v)}')
            axis.set_title(f'symplectomorphicity_condition - max: {np.max(results.global_error_vd["S"]):.3e}\nglobal:blue, local:green')
            #axis.semilogy(results.t_v, S_v)
            axis.semilogy(results.t_v, results.global_error_vd['S'], '.', color='blue')
            axis.semilogy(results.t_v, results.local_error_vd['S'], '.', color='green')

            axis = axis_vv[0][3]
            axis.set_title(f'singular values - max abs: {np.max(np.abs(svd_t))}')
            axis.semilogy(results.t_v, svd_t)

            axis = axis_vv[1][3]
            axis.set_title(f'abs(Lyapunov exponents) computed from singular values - max: {np.max(svd_lyapunov_exponent_t[-1])}')
            axis.semilogy(results.t_v, svd_lyapunov_exponent_t)

            #axis = axis_vv[0][3]
            #axis.set_title('eigenvalues')
            #axis.semilogy(results.t_v, eigenvalues_t[:,0], color='blue')
            #axis.semilogy(results.t_v, eigenvalues_t[:,1], color='green')

            #axis = axis_vv[1][3]
            #axis.set_title('abs(Lyapunov exponents) computed from eigenvalues')
            #axis.semilogy(results.t_v, np.abs(eigenvalues_lyapunov_exponent_v[:,0]), color='blue')
            #axis.semilogy(results.t_v, np.abs(eigenvalues_lyapunov_exponent_v[:,1]), color='green')

            axis = axis_vv[0][4]
            axis.set_title(f't_step_iteration_count - max: {np.max(results.t_step_iteration_count_v)}, mean: {np.mean(results.t_step_iteration_count_v)}')
            axis.semilogy(results.t_v[:-1], results.t_step_iteration_count_v, '.', alpha=0.2)

            axis = axis_vv[1][4]
            axis.set_title(f'J condition number - max: {np.max(J_condition_number_v)}')
            axis.semilogy(results.t_v, J_condition_number_v, '.', alpha=0.2)

            print('\n\n')

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

    def plot_pendulum_dynamics ():
        q = sp.var('q')
        p = sp.var('p')
        # These are Darboux coordinates on T^{*} R
        qp = np.array([q, p])

        # The goal is to find an initial condition which almost loops over the top, but doesn't.
        # if q = pi-epsilon and p = 0, then H = 0 - cos(pi-epsilon) = -(-1+epsilon) = 1 - epsilon,
        # so solving for the kinetic energy for when q = 0,
        #    1 - epsilon = H = p**2/2 - cos(0) = p**2/2 - 1
        # => 2 - epsilon = p**2/2
        # => 4 - 2*epsilon = p**2
        # => 2 - sqrt(2*epsilon) = p

        H = p**2/2 - sp.cos(q)
        X = vorpy.symplectic.symplectic_gradient_of(H, qp)
        DX = vorpy.symbolic.differential(X, qp)

        # In the 1-dimensional configuration space case, it just requires that the transformation preserves area.
        J = vorpy.symbolic.tensor('J', (2,2))
        # Make the symplectomorphicity_condition a scalar.
        S = sp.sqrt(np.sum(np.square(vorpy.symplectic.symplectomorphicity_condition(J, dtype=sp.Integer, return_as_scalar_if_possible=True)))).simplify()

        print(f'H = {H}')
        print(f'X = {X}')
        print(f'DX = {DX}')

        replacement_d = {
            'array':'np.array',
            'cos':'np.cos',
            'sin':'np.sin',
            'sqrt':'np.sqrt',
            'dtype=object':'dtype=float',
        }
        X_fast = vorpy.symbolic.lambdified(X, qp, replacement_d=replacement_d, verbose=True)
        DX_fast = vorpy.symbolic.lambdified(DX, qp, replacement_d=replacement_d, verbose=True)
        H_fast = vorpy.symbolic.lambdified(H, qp, replacement_d=replacement_d, verbose=True)
        S_fast = vorpy.symbolic.lambdified(S, J, replacement_d=replacement_d, verbose=True)

        t_initial = 0.0
        t_final = 200.0

        curve_count = 5
        y_initial_v = [np.array([0.0, p_initial]) for p_initial in np.linspace(0.0, 1.999, curve_count+1)[1:]] # Skip 0

        apply_along_y_t_axes = (1,)
        apply_along_J_t_axes = (1,2)

        plot_dynamics(pathlib.Path('pendulum.png'), t_initial, t_final, y_initial_v, X_fast, DX_fast, H_fast, S_fast, apply_along_y_t_axes, apply_along_J_t_axes)

    def plot_double_pendulum_dynamics ():
        """
        The double pendulum is supposed to be chaotic.

        TODO: Find reference and look up what the expected Lyapunov exponents are
        """

        q = vorpy.symbolic.tensor('q', (2,))
        p = vorpy.symbolic.tensor('p', (2,))
        # These are Darboux coordinates on T^{*} R
        qp = np.array([q, p])

        arm0_endpoint       = np.array([sp.sin(q[0]), -sp.cos(q[0])])
        arm1_endpoint       = np.array([sp.sin(q[1]), -sp.cos(q[1])]) + arm0_endpoint
        arm_endpoint        = np.array([arm0_endpoint, arm1_endpoint])

        arm0_center_of_mass = arm0_endpoint/2
        arm1_center_of_mass = np.array([sp.sin(q[1]), -sp.cos(q[1])])/2 + arm0_endpoint

        arm_center_of_mass  = np.array([arm0_center_of_mass, arm1_center_of_mass])
        #arm_velocity        = np.array(sp.Subs(vorpy.symbolic.differential(arm_center_of_mass, q), q, p).doit().tolist())
        arm_velocity        = vorpy.tensor.contract('ijk,k', vorpy.symbolic.differential(arm_center_of_mass, q), p, dtype=object)
        print(f'arm velocity:\n{arm_velocity}')

        # Assume unit mass and unit moment of inertia
        K = np.sum(np.square(arm_velocity))/2 + np.sum(np.square(p))/2
        U = np.sum(arm_center_of_mass[:,1]) # y values give potential energy
        H = K + U
        X = vorpy.symplectic.symplectic_gradient_of(H, qp)
        DX = vorpy.symbolic.differential(X, qp)

        # Phase space has shape (2,2), so if F is a time-t flow map, then DF is a matrix with shape (2,2,2,2).
        J = vorpy.symbolic.tensor('J', (2,2,2,2))
        # Make the symplectomorphicity_condition a scalar.
        S = sp.sqrt(np.sum(np.square(vorpy.symplectic.symplectomorphicity_condition(J, dtype=sp.Integer, return_as_scalar_if_possible=True)))).simplify()

        print(f'H = {H}')
        print(f'X = {X}')
        print(f'DX = {DX}')

        replacement_d = {
            'array':'np.array',
            'cos':'np.cos',
            'sin':'np.sin',
            'sqrt':'np.sqrt',
            'dtype=object':'dtype=float',
        }
        X_fast = vorpy.symbolic.lambdified(X, qp, replacement_d=replacement_d, verbose=True)
        DX_fast = vorpy.symbolic.lambdified(DX, qp, replacement_d=replacement_d, verbose=True)
        H_fast = vorpy.symbolic.lambdified(H, qp, replacement_d=replacement_d, verbose=True)
        S_fast = vorpy.symbolic.lambdified(S, J, replacement_d=replacement_d, verbose=True)

        print('arm_endpoint_fast:')
        arm_endpoint_fast = vorpy.symbolic.lambdified(arm_endpoint, qp, replacement_d=replacement_d, verbose=True)
        print()

        t_initial = 0.0
        t_final = 20.0

        curve_count = 5
        y_initial_v = [np.array([[0.0, 0.0], [0.0, p_initial]]) for p_initial in np.linspace(0.0, 3.0, curve_count+1)[1:]] # Skip 0

        apply_along_y_t_axes = (1,2)
        apply_along_J_t_axes = (1,2,3,4)

        def plot_function (axis, results):
            axis.set_title('pendulum positions in euclidean space\narm 0 endpoint is blue, arm 1 endpoint is green')
            axis.set_aspect(1.0)
            arm_endpoint_t = vorpy.apply_along_axes(arm_endpoint_fast, apply_along_y_t_axes, (results.y_t,))
            axis.plot(arm_endpoint_t[:,0,0], arm_endpoint_t[:,0,1], color='blue')
            axis.plot(arm_endpoint_t[:,1,0], arm_endpoint_t[:,1,1], color='green')

        plot_dynamics(pathlib.Path('double-pendulum.png'), t_initial, t_final, y_initial_v, X_fast, DX_fast, H_fast, S_fast, apply_along_y_t_axes, apply_along_J_t_axes, plot_function_o=plot_function)

    def plot_kepler_dynamics ():
        q = vorpy.symbolic.tensor('q', (2,))
        p = vorpy.symbolic.tensor('p', (2,))
        # These are Darboux coordinates on T^{*} R
        qp = np.array([q, p])

        H = np.sum(np.square(p))/2 - 1/sp.sqrt(np.sum(np.square(q)))
        X = vorpy.symplectic.symplectic_gradient_of(H, qp)
        DX = vorpy.symbolic.differential(X, qp)

        # Phase space has shape (2,2), so if F is a time-t flow map, then DF is a matrix with shape (2,2,2,2).
        J = vorpy.symbolic.tensor('J', (2,2,2,2))
        # Make the symplectomorphicity_condition a scalar.
        S = sp.sqrt(np.sum(np.square(vorpy.symplectic.symplectomorphicity_condition(J, dtype=sp.Integer, return_as_scalar_if_possible=True)))).simplify()

        print(f'H = {H}')
        print(f'X = {X}')
        print(f'DX = {DX}')

        replacement_d = {
            'array':'np.array',
            'cos':'np.cos',
            'sin':'np.sin',
            'sqrt':'np.sqrt',
            'dtype=object':'dtype=float',
        }
        X_fast = vorpy.symbolic.lambdified(X, qp, replacement_d=replacement_d, verbose=True)
        DX_fast = vorpy.symbolic.lambdified(DX, qp, replacement_d=replacement_d, verbose=True)
        H_fast = vorpy.symbolic.lambdified(H, qp, replacement_d=replacement_d, verbose=True)
        S_fast = vorpy.symbolic.lambdified(S, J, replacement_d=replacement_d, verbose=True)

        t_initial = 0.0
        t_final = 100.0

        curve_count = 5
        y_initial_v = [np.array([[1.0, 0.0], [0.0, p_initial]]) for p_initial in np.linspace(0.0, 1.0, curve_count+1)[1:]] # Skip 0

        apply_along_y_t_axes = (1,2)
        apply_along_J_t_axes = (1,2,3,4)

        plot_dynamics(pathlib.Path('kepler.png'), t_initial, t_final, y_initial_v, X_fast, DX_fast, H_fast, S_fast, apply_along_y_t_axes, apply_along_J_t_axes)

    def plot_kepler_heisenberg_dynamics ():
        x, y, z, p_x, p_y, p_z = sp.var('x, y, z, p_x, p_y, p_z')
        q = np.array([x, y, z])
        p = np.array([p_x, p_y, p_z])
        # These are Darboux coordinates on T^{*} R^3
        qp = np.array([q, p])
        vorpy.symplectic.validate_darboux_coordinates_quantity_or_raise(qp)

        P_x = p_x - y*p_z/2
        P_y = p_y + x*p_z/2
        K = (P_x**2 + P_y**2)/2
        r_squared = x**2 + y**2
        U = -1 / (8*sp.pi*sp.sqrt(r_squared**2 + 16*z**2))

        H = K + U

        X = vorpy.symplectic.symplectic_gradient_of(H, qp)
        DX = vorpy.symbolic.differential(X, qp)

        # Phase space has shape (2,3), so if F is a time-t flow map, then DF is a matrix with shape (2,3,2,3).
        J = vorpy.symbolic.tensor('J', qp.shape+qp.shape)
        # Make the symplectomorphicity_condition a scalar.
        S_cond = vorpy.symplectic.symplectomorphicity_condition(J, dtype=sp.Integer, return_as_scalar_if_possible=True)
        S = sp.sqrt(np.sum(np.square(S_cond))).simplify()

        print(f'H = {H}')
        print(f'X = {X}')
        print(f'DX = {DX}')

        replacement_d = {
            'array':'np.array',
            'cos':'np.cos',
            'sin':'np.sin',
            'sqrt':'np.sqrt',
            'pi':'np.pi',
            'dtype=object':'dtype=float',
        }
        X_fast = vorpy.symbolic.lambdified(X, qp, replacement_d=replacement_d, verbose=True)
        DX_fast = vorpy.symbolic.lambdified(DX, qp, replacement_d=replacement_d, verbose=True)
        H_fast = vorpy.symbolic.lambdified(H, qp, replacement_d=replacement_d, verbose=True)
        S_cond_fast = vorpy.symbolic.lambdified(S_cond, J, replacement_d=replacement_d, verbose=True)
        S_fast = vorpy.symbolic.lambdified(S, J, replacement_d=replacement_d, verbose=True)

        t_initial = 0.0
        t_final = 50.0
        #t_final = 1000.0

        def plot_function (axis, results):
            axis.set_title('(x(t), y(t))')
            axis.set_aspect(1.0)
            axis.plot(results.y_t[:,0,0], results.y_t[:,0,1])
            axis.plot([0.0], [0.0], '.', color='black')

            #S_cond_t = vorpy.apply_along_axes(S_cond_fast, apply_along_J_t_axes, (results.J_t,))
            #axis.set_title('S_cond')
            #axis.plot(results.t_v, S_cond_t.reshape(len(results.t_v), -1))

            #max_abs_S_cond_v = vorpy.apply_along_axes(lambda x:np.max(np.abs(S_cond_fast(x))), apply_along_J_t_axes, (results.J_t,))
            #overall_max = np.max(max_abs_S_cond_v)
            #axis.set_title(f'max abs S_cond - max over all time: {overall_max}')
            #axis.semilogy(results.t_v, max_abs_S_cond_v)

            #axis.set_title('time step size')
            #axis.semilogy(results.t_v[:-1], results.t_step_v, '.', alpha=0.1)

        def plot_function_2 (axis, results):
            axis.set_title('(t, z(t))')
            axis.plot(results.t_v, results.y_t[:,0,2])
            axis.axhline(0.0, color='black')

        #H_initial_v = [sp.Rational(n,4) for n in range(0,2+1)]
        ##H_initial_v = [sp.Rational(n,4) for n in range(-2,2+1)]

        #x_initial_v = [float(sp.Rational(n,8) + 1) for n in range(-2,2+1)]
        #assert 1.0 in x_initial_v # We want exactly 1 to be in this.

        #p_x_initial_v = [float(sp.Rational(n,16)) for n in range(-2,2+1)]
        #assert 0.0 in p_x_initial_v # We want exactly 0 to be in this.

        #p_theta_initial_v = np.linspace(0.05, 0.4, 3)

        H_initial_v = [sp.Integer(0)]

        x_initial_v = [1.0]
        assert 1.0 in x_initial_v # We want exactly 1 to be in this.

        p_x_initial_v = [float(sp.Rational(n,16)) for n in range(-3,3+1)]
        assert 0.0 in p_x_initial_v # We want exactly 0 to be in this.

        p_theta_initial_v = np.linspace(0.05, 0.4, 10)

        for H_initial in H_initial_v:

            # For now, we want to pick an initial condition where H == 0, so solve symbolically for p_z.  Just
            # use sheet_index == 0.
            sheet_index = 0
            p_z_solution_v = sp.solve(H - H_initial, p_z)
            print(f'There are {len(p_z_solution_v)} solutions for the equation: {H} = {H_initial}')
            for i,p_z_solution in enumerate(p_z_solution_v):
                print(f'    solution {i}: p_z = {p_z_solution}')
            # Take the solution specified by sheet_index
            p_z_solution = p_z_solution_v[sheet_index]
            print(f'using solution {sheet_index}: {p_z_solution}')

            p_z_solution_fast = vorpy.symbolic.lambdified(p_z_solution, qp, replacement_d=replacement_d, verbose=True)

            for x_initial,p_x_initial,p_theta_initial in itertools.product(x_initial_v, p_x_initial_v, p_theta_initial_v):
                # Using the symmetry arguments in KH paper, the initial conditions can be constrained.
                y_initial = np.array([[x_initial, 0.0, 0.0], [p_x_initial, p_theta_initial, np.nan]])
                p_z_initial = p_z_solution_fast(y_initial)
                print(f'p_z_initial = {p_z_initial}')
                y_initial[1,2] = p_z_initial
                print(f'y_initial:\n{y_initial}')

                apply_along_y_t_axes = (1,2)
                apply_along_J_t_axes = (1,2,3,4)

                plot_p = pathlib.Path('kh.06.cartesian') / f'H={float(H_initial)}.x={x_initial}.p_x={p_x_initial}.p_theta={p_theta_initial}.t_final={t_final}.png'
                plot_dynamics(plot_p, t_initial, t_final, [y_initial], X_fast, DX_fast, H_fast, S_fast, apply_along_y_t_axes, apply_along_J_t_axes, plot_function_o=plot_function, plot_function_2_o=plot_function_2, write_pickle=True)

    def plot_kepler_heisenberg_dynamics_stretched_cylindrical ():
        R, theta, z, p_R, p_theta, p_z = sp.var('R, theta, z, p_R, p_theta, p_z')
        q = np.array([R, theta, z])
        p = np.array([p_R, p_theta, p_z])
        # These are Darboux coordinates on T^{*} R^3
        qp = np.array([q, p])
        vorpy.symplectic.validate_darboux_coordinates_quantity_or_raise(qp)

        P_R = sp.exp(-R)*p_R
        P_theta = sp.exp(-R)*p_theta + sp.exp(R)*p_z/2
        K = (P_R**2 + P_theta**2)/2
        U = -1 / (8*sp.pi*sp.sqrt(sp.exp(4*R) + 16*z**2))

        H = K + U

        X = vorpy.symplectic.symplectic_gradient_of(H, qp)
        DX = vorpy.symbolic.differential(X, qp)

        # Change of coordinates to cartesian
        r = sp.exp(R)
        x = r*sp.cos(theta)
        y = r*sp.sin(theta)

        cartesian_xy = np.array([x, y])

        # Phase space has shape (2,3), so if F is a time-t flow map, then DF is a matrix with shape (2,3,2,3).
        J = vorpy.symbolic.tensor('J', qp.shape+qp.shape)
        # Make the symplectomorphicity_condition a scalar.
        S_cond = vorpy.symplectic.symplectomorphicity_condition(J, dtype=sp.Integer, return_as_scalar_if_possible=True)
        S = sp.sqrt(np.sum(np.square(S_cond))).simplify()

        print(f'H = {H}')
        print(f'X = {X}')
        print(f'DX = {DX}')

        replacement_d = {
            'array':'np.array',
            'cos':'np.cos',
            'sin':'np.sin',
            'sqrt':'np.sqrt',
            'pi':'np.pi',
            'exp':'np.exp',
            'log':'np.log',
            'dtype=object':'dtype=float',
        }
        X_fast = vorpy.symbolic.lambdified(X, qp, replacement_d=replacement_d, verbose=True)
        DX_fast = vorpy.symbolic.lambdified(DX, qp, replacement_d=replacement_d, verbose=True)
        H_fast = vorpy.symbolic.lambdified(H, qp, replacement_d=replacement_d, verbose=True)
        S_cond_fast = vorpy.symbolic.lambdified(S_cond, J, replacement_d=replacement_d, verbose=True)
        S_fast = vorpy.symbolic.lambdified(S, J, replacement_d=replacement_d, verbose=True)
        cartesian_xy_fast = vorpy.symbolic.lambdified(cartesian_xy, qp, replacement_d=replacement_d, verbose=True)

        t_initial = 0.0
        t_final = 50.0
        #t_final = 1000.0

        def plot_function (axis, results):
            xy_t = vorpy.apply_along_axes(cartesian_xy_fast, (1,2), (results.y_t,))

            axis.set_title('(x(t), y(t))')
            axis.set_aspect(1.0)
            axis.plot(xy_t[:,0], xy_t[:,1])
            axis.plot([0.0], [0.0], '.', color='black')

            #S_cond_t = vorpy.apply_along_axes(S_cond_fast, apply_along_J_t_axes, (results.J_t,))
            #axis.set_title('S_cond')
            #axis.plot(results.t_v, S_cond_t.reshape(len(results.t_v), -1))

            #max_abs_S_cond_v = vorpy.apply_along_axes(lambda x:np.max(np.abs(S_cond_fast(x))), apply_along_J_t_axes, (results.J_t,))
            #overall_max = np.max(max_abs_S_cond_v)
            #axis.set_title(f'max abs S_cond - max over all time: {overall_max}')
            #axis.semilogy(results.t_v, max_abs_S_cond_v)

            #axis.set_title('time step size')
            #axis.semilogy(results.t_v[:-1], results.t_step_v, '.', alpha=0.1)

        def plot_function_2 (axis, results):
            axis.set_title(f'(t, z(t))\nfailure_explanation_o = {results.failure_explanation_o}')
            axis.plot(results.t_v, results.y_t[:,0,2])
            axis.axhline(0.0, color='black')

        #H_initial_v = [sp.Rational(n,4) for n in range(0,2+1)]
        H_initial_v = [sp.Integer(0)]
        #H_initial_v = [sp.Rational(n,4) for n in range(-2,2+1)]

        #R_initial_v = [sp.log(float(sp.Rational(n,8) + 1)) for n in range(-2,2+1)]
        R_initial_v = [0.0]
        assert 0.0 in R_initial_v # We want exactly 0 to be in this.

        p_R_initial_v = [float(sp.Rational(n,16)) for n in range(-3,3+1)]
        #p_R_initial_v = [0.0]
        assert 0.0 in p_R_initial_v # We want exactly 0 to be in this.

        p_theta_initial_v = np.linspace(0.05, 0.4, 10)

        for H_initial in H_initial_v:

            # For now, we want to pick an initial condition where H == 0, so solve symbolically for p_z.  Just
            # use sheet_index == 0.
            sheet_index = 0
            p_z_solution_v = sp.solve(H - H_initial, p_z)
            print(f'There are {len(p_z_solution_v)} solutions for the equation: {H} = {H_initial}')
            for i,p_z_solution in enumerate(p_z_solution_v):
                print(f'    solution {i}: p_z = {p_z_solution}')
            # Take the solution specified by sheet_index
            p_z_solution = p_z_solution_v[sheet_index]
            print(f'using solution {sheet_index}: {p_z_solution}')

            p_z_solution_fast = vorpy.symbolic.lambdified(p_z_solution, qp, replacement_d=replacement_d, verbose=True)

            for R_initial,p_R_initial,p_theta_initial in itertools.product(R_initial_v, p_R_initial_v, p_theta_initial_v):
                # Using the symmetry arguments in KH paper, the initial conditions can be constrained.
                y_initial = np.array([[R_initial, 0.0, 0.0], [p_R_initial, p_theta_initial, np.nan]])
                p_z_initial = p_z_solution_fast(y_initial)
                print(f'p_z_initial = {p_z_initial}')
                y_initial[1,2] = p_z_initial
                print(f'y_initial:\n{y_initial}')

                apply_along_y_t_axes = (1,2)
                apply_along_J_t_axes = (1,2,3,4)

                plot_p = pathlib.Path('kh.06.stretchedcylindrical') / f'H={float(H_initial)}.R={R_initial}.p_R={p_R_initial}.p_theta={p_theta_initial}.t_final={t_final}.png'
                plot_dynamics(plot_p, t_initial, t_final, [y_initial], X_fast, DX_fast, H_fast, S_fast, apply_along_y_t_axes, apply_along_J_t_axes, plot_function_o=plot_function, plot_function_2_o=plot_function_2, write_pickle=True)

    #plot_pendulum_dynamics()
    #plot_double_pendulum_dynamics()
    #plot_kepler_dynamics()
    #plot_kepler_heisenberg_dynamics()
    plot_kepler_heisenberg_dynamics_stretched_cylindrical()
