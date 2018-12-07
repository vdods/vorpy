import matplotlib.pyplot as plt
import numpy as np
import pathlib
import scipy.interpolate
import vorpy.experimental.kh
import vorpy.realfunction.piecewiselinear

def plot_J_equal_zero_extrapolated_trajectory (p_y_initial:float) -> None:
    x_initial = 1.0
    y_initial = 0.0
    z_initial = 0.0
    p_x_initial = 0.0
    H_initial = 0.0
    solution_sheet = 0

    qp_initial = vorpy.experimental.kh.EuclideanNumerics.qp_constrained_by_H__fast(
        np.array([x_initial, y_initial, z_initial, p_x_initial, p_y_initial, H_initial])
    )[solution_sheet]

    p_z_initial = qp_initial[1,2]

    pickle_file_p = pathlib.Path(f'kh_dilation.temp/qp.p_y={p_y_initial}.pickle')
    pickle_file_p.parent.mkdir(parents=True, exist_ok=True)

    results = vorpy.experimental.kh.EuclideanNumerics.compute_trajectory(
        pickle_file_p,
        qp_initial,
        t_final=15.0,
        solution_sheet=0,
    )

    #
    # Try to construct the t < 0 portion of the solution.
    #

    # Solve for the smallest P > 0 such that z(P) == 0.

    t_v = results.t_v
    qp_v = results.y_t
    z_v = qp_v[:,0,2]

    #print(z_v)

    z_zero_index_pair_v, z_zero_orientation_v, z_zero_t_v = vorpy.realfunction.piecewiselinear.oriented_zeros(z_v, t_v=t_v)
    #print(np.vstack((z_zero_index_pair_v.T, z_zero_orientation_v, z_zero_t_v)))
    assert len(z_zero_t_v) >= 2
    assert z_zero_t_v[0] == t_v[0]

    P_bound_index = z_zero_index_pair_v[1,1]
    P = z_zero_t_v[1]
    assert P <= t_v[P_bound_index]

    qp_interpolator = scipy.interpolate.interp1d(t_v, qp_v, axis=0)
    qp_P = qp_interpolator(P)

    segment_t_v = -t_v[:P_bound_index+1]
    segment_qp_v = np.copy(qp_v[:P_bound_index+1,...])

    # Reverse these so that time goes forward.
    segment_t_v[...] = segment_t_v[::-1]
    segment_qp_v[...] = segment_qp_v[::-1,...]

    # Apply transformations to segment_qp_v (flip y, z, p_y, p_z, then flip p_x, p_y, p_z,
    # which is equivalent to flipping y, z, p_x).
    segment_qp_v[:,0,1] *= -1
    segment_qp_v[:,0,2] *= -1
    segment_qp_v[:,1,0] *= -1

    segment_qp_interpolator = scipy.interpolate.interp1d(segment_t_v, segment_qp_v, axis=0)
    segment_qp_minus_P = segment_qp_interpolator(-P)

    angle_v = np.arctan2(qp_P[:,1], qp_P[:,0]) - np.arctan2(segment_qp_minus_P[:,1], segment_qp_minus_P[:,0])
    for i in range(len(angle_v)):
        while angle_v[i] < 0:
            angle_v[i] += 2*np.pi

    print(f'angle_v = {angle_v}')
    if np.max(angle_v) - np.min(angle_v) < 1.0e-6:
        print(f'angles matched as expected')
    else:
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!\n! ANGLES DID NOT MATCH !\n!!!!!!!!!!!!!!!!!!!!!!!!')

    angle = np.mean(angle_v)

    R = np.array([
        [ np.cos(-angle), np.sin(-angle), 0.0],
        [-np.sin(-angle), np.cos(-angle), 0.0],
        [            0.0,            0.0, 1.0],
    ])
    extrapolated_segment_qp_v = np.einsum('ij,tpj->tpi', R, segment_qp_v)
    interpolated_segment_t_v = segment_t_v+2*P
    interpolated_segment_qp_v = qp_interpolator(interpolated_segment_t_v)

    extrapolation_error = np.max(np.abs(extrapolated_segment_qp_v - interpolated_segment_qp_v))

    print(f'extrapolation_error = {extrapolation_error}')
    if extrapolation_error > 5.0e-5:
        print('!!!!!!!!!!!!!!!!!!\n! ERROR EXCEEDED !\n!!!!!!!!!!!!!!!!!!')

    row_count   = 2
    col_count   = 2
    size        = 6
    fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

    for phase_index in range(2):
        s = 'p_' if phase_index == 1 else ''
        angle = angle_v[phase_index]

        axis = axis_vv[phase_index][0]
        axis.set_title(f'initial ({s}x, {s}y) = {(p_x_initial, p_y_initial)}\n({s}x(t), {s}y(t))\nblue:solution, green:extrapolated\nangle = {angle}')
        axis.set_aspect(1.0)

        #axis.plot(qp_v[0:1,phase_index,0], qp_v[0:1,phase_index,1], 'o', color='blue', alpha=0.5)
        axis.plot(qp_v[:,phase_index,0], qp_v[:,phase_index,1], color='blue')
        axis.plot(qp_P[phase_index,0], qp_P[phase_index,1], 'o', color='blue', alpha=0.5)
        axis.plot([0, qp_P[phase_index,0]], [0, qp_P[phase_index,1]], color='black', alpha=0.5)

        #axis.plot(segment_qp_v[0:1,phase_index,0], segment_qp_v[0:1,phase_index,1], 'o', color='green', alpha=0.5)
        axis.plot(segment_qp_v[:,phase_index,0], segment_qp_v[:,phase_index,1], color='green')
        axis.plot(segment_qp_minus_P[phase_index,0], segment_qp_minus_P[phase_index,1], 'o', color='green', alpha=0.5)
        axis.plot([0, segment_qp_minus_P[phase_index,0]], [0, segment_qp_minus_P[phase_index,1]], color='black', alpha=0.5)

        axis.plot(extrapolated_segment_qp_v[:,phase_index,0], extrapolated_segment_qp_v[:,phase_index,1], color='orange', alpha=0.5)

        axis = axis_vv[phase_index][1]
        axis.set_title(f'initial {s}z = {p_z_initial}\n(t, {s}z(t))\nblue:solution, green:extrapolated\nmax extrapolation error = {extrapolation_error}')

        axis.plot(t_v, qp_v[:,phase_index,2], color='blue')
        axis.plot(segment_t_v, segment_qp_v[:,phase_index,2], color='green')
        axis.plot(interpolated_segment_t_v, extrapolated_segment_qp_v[:,phase_index,2], color='orange', alpha=0.5)


    plot_p = pathlib.Path(f'kh_dilation.temp/qp.p_y={p_y_initial}.png')

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

def plot_dilating_extrapolated_trajectory (p_y_initial:float, lam:float) -> None:
    x_initial = 1.0
    y_initial = 0.0
    z_initial = 0.0
    p_x_initial = 0.0
    H_initial = 0.0
    solution_sheet = 0

    qp_initial = vorpy.experimental.kh.EuclideanNumerics.qp_constrained_by_H__fast(
        np.array([x_initial, y_initial, z_initial, p_x_initial, p_y_initial, H_initial])
    )[solution_sheet]

    pickle_file_p = pathlib.Path(f'kh_dilation.temp/qp.p_y={p_y_initial}.pickle')
    pickle_file_p.parent.mkdir(parents=True, exist_ok=True)

    results = vorpy.experimental.kh.EuclideanNumerics.compute_trajectory(
        pickle_file_p,
        qp_initial,
        t_final=15.0,
        solution_sheet=0,
    )

    #
    # Try to construct the t < 0 portion of the solution.
    #

    # Solve for the smallest P > 0 such that z(P) == 0 and P is a positively oriented zero of z..

    t_v = results.t_v
    qp_v = results.y_t
    z_v = qp_v[:,0,2]

    #print(z_v)

    z_zero_index_pair_v, z_zero_orientation_v, z_zero_t_v = vorpy.realfunction.piecewiselinear.oriented_zeros(z_v, t_v=t_v, orientation_p=(lambda o:o < 0))
    #print(np.vstack((z_zero_index_pair_v.T, z_zero_orientation_v, z_zero_t_v)))
    assert len(z_zero_t_v) >= 2
    assert z_zero_t_v[0] == t_v[0]

    P_bound_index = z_zero_index_pair_v[1,1]
    P = z_zero_t_v[1]
    assert P <= t_v[P_bound_index]

    qp_interpolator = scipy.interpolate.interp1d(t_v, qp_v, axis=0)
    qp_P = qp_interpolator(P)

    segment_t_v = -t_v[:P_bound_index+1]
    segment_qp_v = np.copy(qp_v[:P_bound_index+1,...])

    # Reverse these so that time goes forward.
    segment_t_v[...] = segment_t_v[::-1]
    segment_qp_v[...] = segment_qp_v[::-1,...]

    # Apply transformations to segment_qp_v (flip y, z, p_y, p_z, then flip p_x, p_y, p_z,
    # which is equivalent to flipping y, z, p_x).
    segment_qp_v[:,0,1] *= -1
    segment_qp_v[:,0,2] *= -1
    segment_qp_v[:,1,0] *= -1

    segment_qp_interpolator = scipy.interpolate.interp1d(segment_t_v, segment_qp_v, axis=0)
    segment_qp_minus_P = segment_qp_interpolator(-P)

    angle_v = np.arctan2(qp_P[:,1], qp_P[:,0]) - np.arctan2(segment_qp_minus_P[:,1], segment_qp_minus_P[:,0])
    for i in range(len(angle_v)):
        while angle_v[i] < 0:
            angle_v[i] += 2*np.pi

    print(f'angle_v = {angle_v}')
    if np.max(angle_v) - np.min(angle_v) < 1.0e-6:
        print(f'angles matched as expected')
    else:
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!\n! ANGLES DID NOT MATCH !\n!!!!!!!!!!!!!!!!!!!!!!!!')

    angle = np.mean(angle_v)

    R = np.array([
        [ np.cos(-angle), np.sin(-angle), 0.0],
        [-np.sin(-angle), np.cos(-angle), 0.0],
        [            0.0,            0.0, 1.0],
    ])
    extrapolated_segment_qp_v = np.einsum('ij,tpj->tpi', R, segment_qp_v)
    interpolated_segment_t_v = segment_t_v+2*P
    interpolated_segment_qp_v = qp_interpolator(interpolated_segment_t_v)

    extrapolation_error = np.max(np.abs(extrapolated_segment_qp_v - interpolated_segment_qp_v))

    print(f'extrapolation_error = {extrapolation_error}')
    if extrapolation_error > 5.0e-5:
        print('!!!!!!!!!!!!!!!!!!\n! ERROR EXCEEDED !\n!!!!!!!!!!!!!!!!!!')

    row_count   = 2
    col_count   = 2
    size        = 6
    fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

    for phase_index in range(2):
        s = 'p_' if phase_index == 1 else ''
        angle = angle_v[phase_index]

        axis = axis_vv[phase_index][0]
        axis.set_title(f'initial ({s}x, {s}y) = {(p_x_initial, p_y_initial)}\n({s}x(t), {s}y(t))\nblue:solution, green:extrapolated\nangle = {angle}')
        axis.set_aspect(1.0)

        #axis.plot(qp_v[0:1,phase_index,0], qp_v[0:1,phase_index,1], 'o', color='blue', alpha=0.5)
        axis.plot(qp_v[:,phase_index,0], qp_v[:,phase_index,1], color='blue')
        axis.plot(qp_P[phase_index,0], qp_P[phase_index,1], 'o', color='blue', alpha=0.5)
        axis.plot([0, qp_P[phase_index,0]], [0, qp_P[phase_index,1]], color='black', alpha=0.5)

        #axis.plot(segment_qp_v[0:1,phase_index,0], segment_qp_v[0:1,phase_index,1], 'o', color='green', alpha=0.5)
        axis.plot(segment_qp_v[:,phase_index,0], segment_qp_v[:,phase_index,1], color='green')
        axis.plot(segment_qp_minus_P[phase_index,0], segment_qp_minus_P[phase_index,1], 'o', color='green', alpha=0.5)
        axis.plot([0, segment_qp_minus_P[phase_index,0]], [0, segment_qp_minus_P[phase_index,1]], color='black', alpha=0.5)

        axis.plot(extrapolated_segment_qp_v[:,phase_index,0], extrapolated_segment_qp_v[:,phase_index,1], color='orange', alpha=0.5)

        axis = axis_vv[phase_index][1]
        axis.set_title(f'initial {s}z = {p_z_initial}\n(t, {s}z(t))\nblue:solution, green:extrapolated\nmax extrapolation error = {extrapolation_error}')

        axis.plot(t_v, qp_v[:,phase_index,2], color='blue')
        axis.plot(segment_t_v, segment_qp_v[:,phase_index,2], color='green')
        axis.plot(interpolated_segment_t_v, extrapolated_segment_qp_v[:,phase_index,2], color='orange', alpha=0.5)


    plot_p = pathlib.Path(f'kh_dilation.temp/qp.p_y={p_y_initial}.png')

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
    #for p_y_initial in np.linspace(0.05, 0.4, 20):
        #plot_J_equal_zero_extrapolated_trajectory(p_y_initial)

    p_y_initial = 0.1
    plot_J_equal_zero_extrapolated_trajectory(p_y_initial)

    pass
