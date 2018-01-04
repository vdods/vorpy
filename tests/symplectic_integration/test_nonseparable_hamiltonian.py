import itertools
import numpy as np
import os
from .hamiltons_equations import deviation_form
from .kepler_nd import KeplerNd
from .pendulum_nd import PendulumNd
from .results import Results
import scipy.integrate
import sys
import vorpy.symplectic_integration

TEST_ARTIFACTS_DIR = 'test_artifacts/symplectic_integration/nonseparable_hamiltonian'

def make_filename_in_artifacts_dir (filename):
    os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)
    return os.path.join(TEST_ARTIFACTS_DIR, filename)

CURVE_T_MAX = 30.0

def compute_reference_integral_curve (N, reference_dt):
    reference_t_v = np.arange(0.0, CURVE_T_MAX, reference_dt)
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

    filename = make_filename_in_artifacts_dir('symplectic_integration_comparison.Pendulum{0}d.order:{1}.dt:{2:.2e}.omega:{3:.2e}.png'.format(N, order, dt, omega))
    results.plot(filename)

# This test function takes a while to run.
def test__compare_with_separable_hamiltonian_integrate ():
    for N in [1,2]:
        for dt in [0.02, 0.2]:
            t_v = np.arange(0.0, CURVE_T_MAX, dt)
            # The reference solution is computed via vorpy.symplectic_integration.separable_hamiltonian.integrate
            # using a very small timestep, in order to be considered as good as the "true" solution.
            reference_dt = dt/10.0
            reference_t_v,reference_qp_v = compute_reference_integral_curve(N, reference_dt)
            for order in [2,4]:
                omega = vorpy.symplectic_integration.nonseparable_hamiltonian.heuristic_estimate_for_omega(delta=dt, order=order)
                compare_with_separable_hamiltonian_integrate(N, dt, t_v, order, omega, reference_dt, reference_t_v, reference_qp_v)

def test__salvaged_result ():
    import sys

    def dH_dq_complaining (q, p):
        # Introduce arbitrary error in order to test SalvagedResultException functionality.
        assert np.sum(np.square(q)) > 1.0e-2
        return KeplerNd.dH_dq(q, p)

    dt = 0.01
    t_v = np.arange(0.0, 60.0, dt)
    # R^3 Kepler problem
    qp_0 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.01, 0.0]
    ])
    order = 2
    omega = np.pi/(4*dt)
    assert np.allclose(2*omega*dt, np.pi/2)
    try:
        results = Results()
        qp_v = vorpy.symplectic_integration.nonseparable_hamiltonian.integrate(
            initial_coordinates=qp_0,
            t_v=t_v,
            dH_dq=dH_dq_complaining,
            dH_dp=KeplerNd.dH_dp,
            order=order,
            omega=omega
        )
        assert False, 'did not catch SalvagedResultException as expected'
    except vorpy.symplectic_integration.exceptions.SalvagedResultException as e:
        print('caught SalvagedResultException as expected: {0}'.format(e))
        qp_v = e.salvaged_qp_v
        t_v = e.salvaged_t_v
        assert qp_v.shape[0] == t_v.shape[0]
        H_v = vorpy.apply_along_axes(KeplerNd.H, (-2,-1), (qp_v,), output_axis_v=(), func_output_shape=())
        deviation_form_v = deviation_form(t_v=t_v, qp_v=qp_v, dH_dq=KeplerNd.dH_dq, dH_dp=KeplerNd.dH_dp)
        norm_deviation_form_v = vorpy.apply_along_axes(np.linalg.norm, (-2,-1), (deviation_form_v,), output_axis_v=(), func_output_shape=())
        norm_error_v = np.full(norm_deviation_form_v.shape, np.nan)
        results.add_result('Kepler trajectory', dt, t_v, qp_v, H_v, norm_deviation_form_v, norm_error_v)
        # As you can see in this plot, the energy is not nearly conserved as time goes on, and the norm of the deviation form
        # diverges away from zero as time goes on.
        results.plot(filename=make_filename_in_artifacts_dir('symplectic_integration.separable_hamiltonian.salvaged_result.png'))
