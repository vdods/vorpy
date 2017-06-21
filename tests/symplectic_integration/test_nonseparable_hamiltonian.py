import itertools
import numpy as np
from .hamiltons_equations import deviation_form
from .pendulum_nd import PendulumNd
from .results import Results
import scipy.integrate
import sys
import vorpy.symplectic_integration

def compute_reference_integral_curve (N, reference_dt):
    reference_t_v = np.arange(0.0, 60.0, reference_dt)
    T = len(reference_t_v)

    qp_0 = np.zeros((2,N), dtype=float)
    qp_0[0,0] = np.pi/2.0

    # The separable integrator is using a 10x finer timestep, so that it's an accurate reference.
    reference_qp_v = vorpy.symplectic_integration.separable_hamiltonian.integrate(
        initial_coordinates=qp_0,
        t_v=reference_t_v,
        dK_dp=PendulumNd.dK_dp,
        dV_dq=PendulumNd.dV_dq,
        update_step_coefficients=vorpy.symplectic_integration.separable_hamiltonian.update_step_coefficients.ruth4
    )

    print('reference_qp_v.shape = {0}'.format(reference_qp_v.shape))
    return reference_t_v,reference_qp_v

def compare_with_separable_hamiltonian_integrate (N, dt, t_v, order, omega, reference_dt, reference_t_v, reference_qp_v):
    results = Results()

    sparsified_reference_qp_v = reference_qp_v[::10,:,:]

    def add_result (result_name, dt, t_v, qp_v, *, compute_norm_error=True):
        nonlocal results
        H_v = vorpy.apply_along_axes(PendulumNd.H, (-2,-1), (qp_v,), output_axis_v=(), func_output_shape=())
        deviation_form_v = deviation_form(t_v=t_v, qp_v=qp_v, dH_dq=PendulumNd.dH_dq, dH_dp=PendulumNd.dH_dp)
        norm_deviation_form_v = vorpy.apply_along_axes(np.linalg.norm, (-2,-1), (deviation_form_v,), output_axis_v=(), func_output_shape=())
        if compute_norm_error:
            norm_error_v = vorpy.apply_along_axes(np.linalg.norm, (-2,-1), (qp_v - sparsified_reference_qp_v,), output_axis_v=(), func_output_shape=())
        else:
            norm_error_v = np.full(norm_deviation_form_v.shape, np.nan)
        results.add_result(result_name, dt, t_v, qp_v, H_v, norm_deviation_form_v, norm_error_v)

    qp_0 = np.zeros((2,N), dtype=float)
    qp_0[0,0] = np.pi/2.0

    qp_v = vorpy.symplectic_integration.nonseparable_hamiltonian.integrate(
        initial_coordinates=qp_0,
        t_v=t_v,
        dH_dq=PendulumNd.dH_dq,
        dH_dp=PendulumNd.dH_dp,
        order=order,
        omega=omega
    )
    assert sparsified_reference_qp_v.shape == qp_v.shape

    max_error = np.max(np.abs(sparsified_reference_qp_v - qp_v))
    sys.stderr.write('N = {0}, dt = {1}, order = {2}, omega = {3}, max_error = {4}\n'.format(N, dt, order, omega, max_error))
    #assert max_error < 0.3e-3

    add_result('using separable Hamiltonian integrator', reference_dt, reference_t_v, reference_qp_v, compute_norm_error=False)
    add_result('using nonseparable Hamiltonian integrator', dt, t_v, qp_v)

    filename = 'symplectic_integration_comparison.Pendulum{0}d.order:{1}.dt:{2:.2e}.omega:{3:.2e}.png'.format(N, order, dt, omega)
    results.plot(filename)

def test__compare_with_separable_hamiltonian_integrate ():
    for N in [1,2,3]:
        for dt in [0.002, 0.02, 0.2]:
            t_v = np.arange(0.0, 60.0, dt)
            # The reference solution is computed via vorpy.symplectic_integration.separable_hamiltonian.integrate
            # using a very small timestep, in order to be considered as good as the "true" solution.
            reference_dt = dt/10.0
            reference_t_v,reference_qp_v = compute_reference_integral_curve(N, reference_dt)
            for order in [2,4,6,8]:
                omega = vorpy.symplectic_integration.nonseparable_hamiltonian.heuristic_estimate_for_omega(delta=dt, order=order)
                compare_with_separable_hamiltonian_integrate(N, dt, t_v, order, omega, reference_dt, reference_t_v, reference_qp_v)
