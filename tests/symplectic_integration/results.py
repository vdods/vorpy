import matplotlib.pyplot as plt
import numpy as np
import vorpy

class Results:
    def __init__ (self):
        self.result_name_v = []
        self.result_d = {}

    def add_result (self, result_name, dt, t_v, qp_v, H_v, norm_deviation_form_v, norm_error_v):
        N = qp_v.shape[-1]
        self.result_name_v.append(result_name)
        self.result_d[result_name] = {
            'N'                     :N,
            'can_plot_phase_space'  :N == 1,
            'dt'                    :dt,
            't_v'                   :t_v,
            'qp_v'                  :qp_v,
            'H_v'                   :H_v,
            'norm_deviation_form_v' :norm_deviation_form_v,
            'norm_error_v'          :norm_error_v,
        }
        assert len(self.result_name_v) == len(self.result_d)

    def plot_result (self, result_name, axis_v, *, detrend_Hamiltonian=False):
        result                  = self.result_d[result_name]
        N                       = result['N']
        can_plot_phase_space    = result['can_plot_phase_space']
        dt                      = result['dt']
        t_v                     = result['t_v']
        qp_v                    = result['qp_v']
        H_v                     = result['H_v']
        norm_deviation_form_v   = result['norm_deviation_form_v']
        norm_error_v            = result['norm_error_v']

        can_plot_phase_space = N == 1
        T = qp_v.shape[0]
        assert H_v.shape[0] == T

        H_v_reshaped = H_v.reshape(T, -1)
        if detrend_Hamiltonian:
            # Detrend each H_v, so that the plot is just the deviation from the mean and the min and max are more meaningful
            H_v_reshaped -= np.mean(H_v_reshaped, axis=0)

        norm_deviation_form_v_reshaped = norm_deviation_form_v.reshape(T, -1)
        norm_error_v_reshaped = norm_error_v.reshape(T, -1)

        min_H   = np.min(H_v_reshaped)
        max_H   = np.max(H_v_reshaped)
        range_H = max_H - min_H

        max_norm_deviation_form      = np.max(norm_deviation_form_v_reshaped[1:-1])
        max_norm_error               = np.max(norm_error_v_reshaped)

        axis    = axis_v[0]
        axis.set_title('Hamiltonian{0}; range = {1:.2e}\nmethod: {2}'.format(' (detrended)' if detrend_Hamiltonian else '', range_H, result_name))
        axis.plot(t_v, H_v_reshaped)
        axis.axhline(min_H, color='green')
        axis.axhline(max_H, color='green')

        sqd_v   = vorpy.apply_along_axes(lambda x:np.sum(np.square(x)), (-2,-1), (qp_v - qp_v[0],))
        sqd_v_reshaped = sqd_v.reshape(T, -1)

        axis    = axis_v[1]
        axis.set_title('{0} : sq dist from initial'.format(result_name))
        axis.semilogy(t_v, sqd_v_reshaped)

        axis    = axis_v[2]
        axis.set_title('{0} : norm of deviation form; max = {1:.2e}\ndt = {2:.2e}'.format(result_name, max_norm_deviation_form, dt))
        axis.semilogy(t_v[1:-1], norm_deviation_form_v_reshaped[1:-1])
        axis.axhline(max_norm_deviation_form, color='green')

        axis    = axis_v[3]
        axis.set_title('{0} : norm of error (compare to reference); max = {1:.2e}'.format(result_name, max_norm_error))
        axis.semilogy(t_v, norm_error_v_reshaped)
        axis.axhline(max_norm_error, color='green')

        # We can only directly plot the phase space if the configuration space is 1-dimensional.
        if can_plot_phase_space:
            # This is to get rid of the last axis, which has size 1, and therefore can be canonically reshaped away
            # via the canonical identification between 1-vectors and scalars.
            qp_v_reshaped = qp_v.reshape(T, -1, 2)
            print('qp_v.shape = {0}'.format(qp_v.shape))
            print('qp_v_reshaped.shape = {0}'.format(qp_v_reshaped.shape))

            axis = axis_v[-1]
            axis.set_title('phase space\nmethod: {0}'.format(result_name))
            axis.set_aspect('equal')
            axis.plot(qp_v_reshaped[...,0], qp_v_reshaped[...,1])

    def plot (self, filename, *, detrend_Hamiltonian=False):
        any_can_plot_phase_space = any(result['can_plot_phase_space'] for _,result in self.result_d.items())

        row_count = len(self.result_name_v)
        col_count = 5 if any_can_plot_phase_space else 4
        fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(8*col_count, 8*row_count))

        for result_name,axis_v in zip(self.result_name_v,axis_vv):
            self.plot_result(result_name, axis_v, detrend_Hamiltonian=detrend_Hamiltonian)

        fig.tight_layout()
        plt.savefig(filename)
        print('wrote to file "{0}"'.format(filename))
        plt.close(fig)

