import itertools
import numpy as np
from .pendulum_nd import PendulumNd
from .results import Results
import scipy.integrate
import vorpy.symplectic_integration

def compare_with_separable_hamiltonian_integrate (N):
    results = Results()

    def add_result (result_name, qp_v):
        nonlocal results
        H_v = vorpy.apply_along_axes(PendulumNd.H, (-2,-1), (qp_v,), output_axis_v=(), func_output_shape=())
        results.add_result(result_name, t_v, qp_v, H_v)

    t_v = np.linspace(0.0, 30.0, 1500)
    qp_0 = np.zeros((2,N), dtype=float)
    qp_0[0,0] = np.pi/2.0

    sep = vorpy.symplectic_integration.separable_hamiltonian.integrate(
        initial_coordinates=qp_0,
        t_v=t_v,
        dK_dp=PendulumNd.dK_dp,
        dV_dq=PendulumNd.dV_dq,
        update_step_coefficients=vorpy.symplectic_integration.separable_hamiltonian.update_step_coefficients.ruth4
    )

    nonsep = vorpy.symplectic_integration.nonseparable_hamiltonian.integrate(
        initial_coordinates=qp_0,
        t_v=t_v,
        dH_dq=PendulumNd.dH_dq,
        dH_dp=PendulumNd.dH_dp
    )

    max_error = np.max(np.abs(sep - nonsep))
    assert max_error < 0.3e-3

    add_result('using separable Hamiltonian integrator', sep)
    add_result('using nonseparable Hamiltonian integrator', nonsep)

    filename = 'symplectic_integration_comparison.Pendulum{0}d.png'.format(N)
    results.plot(filename)

def test__compare_with_separable_hamiltonian_integrate ():
    for N in [1,2,3]:
        compare_with_separable_hamiltonian_integrate(N)
