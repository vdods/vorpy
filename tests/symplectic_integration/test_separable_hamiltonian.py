import itertools
import numpy as np
from .pendulum_nd import PendulumNd
from .results import Results
import scipy.integrate
import vorpy
import vorpy.symplectic_integration

def compare_integrator_methods (N):
    results = Results()

    t_v = np.linspace(0.0, 30.0, 1500)
    qp_0 = np.zeros((2,N), dtype=float)
    qp_0[0,0] = np.pi/2.0

    def add_result (result_name, qp_v):
        nonlocal results
        H_v = vorpy.apply_along_axes(PendulumNd.H, (-2,-1), (qp_v,), output_axis_v=(), func_output_shape=())
        results.add_result(result_name, t_v, qp_v, H_v)

    # scipy.integrate.odeint expects the phase space coordinate to have shape (2*N,), not (2,N).
    # This function adapts between those two conventions.
    def X_H_adapted_for_odeint (coordinates_as_1_tensor, *args):
        qp = coordinates_as_1_tensor.reshape(2,-1)      # -1 will cause reshape to infer the value of N.
        return PendulumNd.X_H(qp, *args).reshape(-1)    # -1 will cause reshape to produce a 1-tensor.

    add_result(
        'standard odeint',
        scipy.integrate.odeint(X_H_adapted_for_odeint, qp_0.reshape(-1), t_v).reshape(len(t_v),2,-1)
    )
    add_result(
        'symplectic Euler',
        vorpy.symplectic_integration.separable_hamiltonian.integrate(
            initial_coordinates=qp_0,
            t_v=t_v,
            dK_dp=PendulumNd.dK_dp,
            dV_dq=PendulumNd.dV_dq,
            update_step_coefficients=vorpy.symplectic_integration.separable_hamiltonian.update_step_coefficients.euler1
        )
    )
    add_result(
        'symplectic Verlet',
        vorpy.symplectic_integration.separable_hamiltonian.integrate(
            initial_coordinates=qp_0,
            t_v=t_v,
            dK_dp=PendulumNd.dK_dp,
            dV_dq=PendulumNd.dV_dq,
            update_step_coefficients=vorpy.symplectic_integration.separable_hamiltonian.update_step_coefficients.verlet2
        )
    )
    add_result(
        'symplectic Ruth3',
        vorpy.symplectic_integration.separable_hamiltonian.integrate(
            initial_coordinates=qp_0,
            t_v=t_v,
            dK_dp=PendulumNd.dK_dp,
            dV_dq=PendulumNd.dV_dq,
            update_step_coefficients=vorpy.symplectic_integration.separable_hamiltonian.update_step_coefficients.ruth3
        )
    )
    add_result(
        'symplectic Ruth4',
        vorpy.symplectic_integration.separable_hamiltonian.integrate(
            initial_coordinates=qp_0,
            t_v=t_v,
            dK_dp=PendulumNd.dK_dp,
            dV_dq=PendulumNd.dV_dq,
            update_step_coefficients=vorpy.symplectic_integration.separable_hamiltonian.update_step_coefficients.ruth4
        )
    )

    filename = 'symplectic_integration.Pendulum{0}d.png'.format(N)
    results.plot(filename)

def compare_parallel_initial_conditions (N, shape_of_parallelism):
    rng = np.random.RandomState(42)

    t_v = np.linspace(0.0, 30.0, 1500)
    base_shape = (2,N)
    base_qp_0 = np.zeros(base_shape, dtype=float)
    base_qp_0[0,0] = np.pi/2.0

    qp_0 = rng.randn(*(shape_of_parallelism+base_shape)) + base_qp_0

    results = Results()

    def add_result (result_name, qp_v):
        nonlocal results
        H_v = vorpy.apply_along_axes(PendulumNd.H, (-2,-1), (qp_v,), output_axis_v=(), func_output_shape=())
        results.add_result(result_name, t_v, qp_v, H_v)

    add_result(
        'symplectic Euler',
        vorpy.symplectic_integration.separable_hamiltonian.integrate(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, update_step_coefficients=vorpy.symplectic_integration.separable_hamiltonian.update_step_coefficients.euler1)
    )
    add_result(
        'symplectic Verlet',
        vorpy.symplectic_integration.separable_hamiltonian.integrate(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, update_step_coefficients=vorpy.symplectic_integration.separable_hamiltonian.update_step_coefficients.verlet2)
    )
    add_result(
        'symplectic Ruth3',
        vorpy.symplectic_integration.separable_hamiltonian.integrate(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, update_step_coefficients=vorpy.symplectic_integration.separable_hamiltonian.update_step_coefficients.ruth3)
    )
    add_result(
        'symplectic Ruth4',
        vorpy.symplectic_integration.separable_hamiltonian.integrate(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, update_step_coefficients=vorpy.symplectic_integration.separable_hamiltonian.update_step_coefficients.ruth4)
    )

    filename = 'symplectic_integration_parallel.Pendulum{0}d.shape:{1}.png'.format(N, shape_of_parallelism)
    results.plot(filename)

def test__compare_integrator_methods ():
    for N in [1,2,3]:
        compare_integrator_methods(N)

def test__compare_parallel_initial_conditions ():
    for N,shape_of_parallelism in itertools.product([1,2], [(),(2,),(2,3)]):
        compare_parallel_initial_conditions(N, shape_of_parallelism)

