import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import vorpy
import vorpy.symplectic_integration

class PendulumNd:
    """
    Defines the various geometric-mechanical structures of a spherical pendulum in arbitrary dimension.

    Coordinates are assumed to have shape (2,N), i.e. np.array([q,p]), where q and p are the vector-valued
    pendular angle and momentum respectively.  The angle coordinates are assumed to be normal coordinates
    with origin denoting the downward, stable equilibrium position.  The potential energy function is the
    vertical position of the pendular mass.
    """

    @staticmethod
    def K (p):
        return 0.5*np.sum(np.square(p))

    @staticmethod
    def V (q):
        # np.linalg.norm(q) gives the angle from the vertical axis
        return np.cos(np.linalg.norm(q))

    @staticmethod
    def H (coordinates):
        q = coordinates[0,:]
        p = coordinates[1,:]
        return PendulumNd.K(p) - PendulumNd.V(q)

    @staticmethod
    def dK_dp (p):
        return p

    @staticmethod
    def dV_dq (q):
        norm_q = np.linalg.norm(q)
        return np.sin(norm_q)/norm_q * q

    @staticmethod
    def X_H (coordinates, *args): # args is assumed to be the time coordinate and other ignored args
        q = coordinates[0,:]
        p = coordinates[1,:]
        # This is the symplectic gradient of H.
        return np.array((PendulumNd.dK_dp(p), -PendulumNd.dV_dq(q)))

class Results:
    def __init__ (self):
        self.result_name_v = []
        self.result_d = {}

    def add_result (self, result_name, t_v, qp_v):
        N = qp_v.shape[-1]
        self.result_name_v.append(result_name)
        self.result_d[result_name] = {
            'N'                     :N,
            'can_plot_phase_space'  :N == 1,
            't_v'                   :t_v,
            'qp_v'                  :qp_v,
            'H_v'                   :vorpy.apply_along_axes(PendulumNd.H, (-2,-1), qp_v, output_axis_v=(), func_output_shape=()),
        }
        assert len(self.result_name_v) == len(self.result_d)

    def plot_result (self, result_name, axis_v):
        result                  = self.result_d[result_name]
        N                       = result['N']
        can_plot_phase_space    = result['can_plot_phase_space']
        t_v                     = result['t_v']
        qp_v                    = result['qp_v']
        H_v                     = result['H_v']

        can_plot_phase_space = N == 1
        T = qp_v.shape[0]
        assert H_v.shape[0] == T

        H_v_reshaped = H_v.reshape(T, -1)
        # Detrend each H_v, so that the plot is just the deviation from the mean and the min and max are more meaningful
        H_v_reshaped -= np.mean(H_v_reshaped, axis=0)

        min_H   = np.min(H_v_reshaped)
        max_H   = np.max(H_v_reshaped)
        range_H = max_H - min_H

        axis    = axis_v[0]
        axis.set_title('Hamiltonian (detrended); range = {0:.2e}\nmethod: {1}'.format(range_H, result_name))
        axis.plot(t_v, H_v_reshaped)
        axis.axhline(min_H, color='green')
        axis.axhline(max_H, color='green')

        sqd_v   = vorpy.apply_along_axes(lambda x:np.sum(np.square(x)), (-2,-1), qp_v - qp_v[0])
        sqd_v_reshaped = sqd_v.reshape(T, -1)

        axis    = axis_v[1]
        axis.set_title('{0} : sq dist from initial'.format(result_name))
        axis.semilogy(t_v, sqd_v_reshaped)

        # We can only directly plot the phase space if the configuration space is 1-dimensional.
        if can_plot_phase_space:
            # This is to get rid of the last axis, which has size 1, and therefore can be canonically reshaped away
            # via the canonical identification between 1-vectors and scalars.
            qp_v_reshaped = qp_v.reshape(T, -1, 2)
            print('qp_v.shape = {0}'.format(qp_v.shape))
            print('qp_v_reshaped.shape = {0}'.format(qp_v_reshaped.shape))

            axis = axis_v[2]
            axis.set_title('phase space\nmethod: {0}'.format(result_name))
            axis.set_aspect('equal')
            axis.plot(qp_v_reshaped[...,0], qp_v_reshaped[...,1])

    def plot (self, filename):
        any_can_plot_phase_space = any(result['can_plot_phase_space'] for _,result in self.result_d.items())

        row_count = len(self.result_name_v)
        col_count = 3 if any_can_plot_phase_space else 2
        fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(8*col_count, 8*row_count))

        for result_name,axis_v in zip(self.result_name_v,axis_vv):
            self.plot_result(result_name, axis_v)

        fig.tight_layout()
        plt.savefig(filename)
        print('wrote to file "{0}"'.format(filename))

def compare_integrator_methods (N):
    results = Results()

    t_v = np.linspace(0.0, 30.0, 1500)
    qp_0 = np.zeros((2,N), dtype=float)
    qp_0[0,0] = np.pi/2.0

    # scipy.integrate.odeint expects the phase space coordinate to have shape (2*N,), not (2,N).
    # This function adapts between those two conventions.
    def X_H_adapted_for_odeint (coordinates_as_1_tensor, *args):
        qp = coordinates_as_1_tensor.reshape(2,-1)      # -1 will cause reshape to infer the value of N.
        return PendulumNd.X_H(qp, *args).reshape(-1)    # -1 will cause reshape to produce a 1-tensor.

    results.add_result('standard odeint', t_v, scipy.integrate.odeint(X_H_adapted_for_odeint, qp_0.reshape(-1), t_v).reshape(len(t_v),2,-1))
    results.add_result('symplectic Euler', t_v, vorpy.symplectic_integration.split_hamiltonian.integrate(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=vorpy.symplectic_integration.split_hamiltonian.predefined_method_coefficients.euler))
    results.add_result('symplectic Verlet', t_v, vorpy.symplectic_integration.split_hamiltonian.integrate(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=vorpy.symplectic_integration.split_hamiltonian.predefined_method_coefficients.verlet))
    results.add_result('symplectic Ruth3', t_v, vorpy.symplectic_integration.split_hamiltonian.integrate(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=vorpy.symplectic_integration.split_hamiltonian.predefined_method_coefficients.ruth3))
    results.add_result('symplectic Ruth4', t_v, vorpy.symplectic_integration.split_hamiltonian.integrate(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=vorpy.symplectic_integration.split_hamiltonian.predefined_method_coefficients.ruth4))

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

    results.add_result('symplectic Euler', t_v, vorpy.symplectic_integration.split_hamiltonian.integrate(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=vorpy.symplectic_integration.split_hamiltonian.predefined_method_coefficients.euler))
    results.add_result('symplectic Verlet', t_v, vorpy.symplectic_integration.split_hamiltonian.integrate(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=vorpy.symplectic_integration.split_hamiltonian.predefined_method_coefficients.verlet))
    results.add_result('symplectic Ruth3', t_v, vorpy.symplectic_integration.split_hamiltonian.integrate(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=vorpy.symplectic_integration.split_hamiltonian.predefined_method_coefficients.ruth3))
    results.add_result('symplectic Ruth4', t_v, vorpy.symplectic_integration.split_hamiltonian.integrate(initial_coordinates=qp_0, t_v=t_v, dK_dp=PendulumNd.dK_dp, dV_dq=PendulumNd.dV_dq, coefficients=vorpy.symplectic_integration.split_hamiltonian.predefined_method_coefficients.ruth4))

    filename = 'symplectic_integration_parallel.Pendulum{0}d.shape:{1}.png'.format(N, shape_of_parallelism)
    results.plot(filename)

def test__compare_integrator_methods ():
    for N in [1,2,3]:
        compare_integrator_methods(N)

def test__compare_parallel_initial_conditions ():
    for N,shape_of_parallelism in itertools.product([1,2], [(),(2,),(2,3)]):
        compare_parallel_initial_conditions(N, shape_of_parallelism)

