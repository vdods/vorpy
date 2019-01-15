import itertools
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import scipy.interpolate
import typing
import vorpy
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

    # TODO: Use Bezier interpolation instead
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

    # TODO: Use Bezier interpolation instead
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

    # TODO: Use Bezier interpolation instead
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

    # TODO: Use Bezier interpolation instead
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

def compute_trajectory (p_x_initial:float, p_y_initial:float, base_dir_p:pathlib.Path) -> vorpy.integration.adaptive.IntegrateVectorFieldResults:
    pickle_file_p = base_dir_p / f'qp.p_x={p_x_initial}_p_y={p_y_initial}.pickle'
    pickle_file_p.parent.mkdir(parents=True, exist_ok=True)

    x_initial = 2.0
    y_initial = 0.0
    z_initial = 0.0
    H_initial = 0.0
    solution_sheet = 0

    qp_initial = vorpy.experimental.kh.EuclideanNumerics.qp_constrained_by_H__fast(
        np.array([x_initial, y_initial, z_initial, p_x_initial, p_y_initial, H_initial])
    )[solution_sheet]

    return vorpy.experimental.kh.EuclideanNumerics.compute_trajectory(
        pickle_file_p,
        qp_initial,
        t_final=100.0,
        solution_sheet=0,
    )

def plot_trajectories (results_v:typing.Sequence[vorpy.integration.adaptive.IntegrateVectorFieldResults], base_dir_p:pathlib.Path) -> None:
    row_count   = 2
    col_count   = 3
    size        = 6
    fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

    for results in results_v:
        t_v = results.t_v
        qp_v = results.y_t

        p_x_initial = qp_v[0,1,0]
        p_y_initial = qp_v[0,1,1]

        for phase_index in range(2):
            s = 'p_' if phase_index == 1 else ''

            axis = axis_vv[phase_index][0]
            axis.set_title(f'initial ({s}x, {s}y) = {(qp_v[0,phase_index,0], qp_v[0,phase_index,1])}\n({s}x(t), {s}y(t))')
            axis.set_aspect(1.0)
            axis.plot(qp_v[:,phase_index,0], qp_v[:,phase_index,1])

            axis = axis_vv[phase_index][1]
            axis.set_title(f'initial {s}z = {qp_v[0,phase_index,2]}\n(t, {s}z(t))')
            axis.plot(t_v, qp_v[:,phase_index,2])

        p_theta_v   = vorpy.apply_along_axes(vorpy.experimental.kh.EuclideanNumerics.p_theta__fast, (1,2), (results.y_t,))

        axis = axis_vv[0][2]
        axis.set_title(f'p_theta = {p_theta_v[0]}')
        axis.plot(t_v, p_theta_v)

    plot_p = base_dir_p / f'qp.p_y={p_y_initial}.png'

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

def plot_trajectories_QuadraticCylindrical (results_v:typing.Sequence[vorpy.integration.adaptive.IntegrateVectorFieldResults], plot_p:pathlib.Path) -> None:
    row_count   = 2
    col_count   = 4
    size        = 6
    fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

    for results in results_v:
        t_v         = results.t_v
        qp_v        = results.y_t

        # TODO: Use Bezier interpolation instead
        qp_interpolator = scipy.interpolate.interp1d(t_v, qp_v, axis=0)
        qp_initial  = qp_v[0]

        p_R_initial = qp_initial[1,0]
        H_initial   = vorpy.experimental.kh.QuadraticCylindricalNumerics.H__fast(qp_initial)
        J_initial   = vorpy.experimental.kh.QuadraticCylindricalNumerics.J__fast(qp_initial)

        theta_v     = qp_v[:,0,1]
        # Segment based on local maxima of theta_v
        _, _, theta_local_max_s_v = vorpy.realfunction.piecewiselinear.local_maximizers(theta_v, t_v=t_v)
        print(f'theta_local_max_s_v = {theta_local_max_s_v}')
        #t_delta_v   = np.diff(theta_local_max_s_v)
        #scale_v     = t_delta_v[1:] / t_delta_v[:-1]

        # Find the sector bounds for the (t,R(t)) curve
        # Using local maxes and mins is very approximate.
        _,_,R_local_max_s_v = vorpy.realfunction.piecewiselinear.local_maximizers(qp_v[:,0,0], t_v=t_v)
        _,_,R_local_min_s_v = vorpy.realfunction.piecewiselinear.local_minimizers(qp_v[:,0,0], t_v=t_v)
        print(f'R_local_max_s_v = {R_local_max_s_v}')
        print(f'R_local_min_s_v = {R_local_min_s_v}')
        R_local_max_v = qp_interpolator(R_local_max_s_v)[:,0,0]
        R_local_min_v = qp_interpolator(R_local_min_s_v)[:,0,0]
        print(f'R_local_max_v = {R_local_max_v}')
        print(f'R_local_min_v = {R_local_min_v}')
        R_sector_upper_bound_slope = np.mean(np.diff(R_local_max_v) / np.diff(R_local_max_s_v))
        R_sector_lower_bound_slope = np.mean(np.diff(R_local_min_v) / np.diff(R_local_min_s_v))
        R_sector_middle_slope = np.mean([R_sector_upper_bound_slope, R_sector_lower_bound_slope])

        # Find the sector bounds for the (t,w(t)) curve
        # Using local maxes and mins is very approximate.
        _,_,w_local_max_s_v = vorpy.realfunction.piecewiselinear.local_maximizers(qp_v[:,0,2], t_v=t_v)
        _,_,w_local_min_s_v = vorpy.realfunction.piecewiselinear.local_minimizers(qp_v[:,0,2], t_v=t_v)
        print(f'w_local_max_s_v = {w_local_max_s_v}')
        print(f'w_local_min_s_v = {w_local_min_s_v}')
        w_local_max_v = qp_interpolator(w_local_max_s_v)[:,0,2]
        w_local_min_v = qp_interpolator(w_local_min_s_v)[:,0,2]
        print(f'w_local_max_v = {w_local_max_v}')
        print(f'w_local_min_v = {w_local_min_v}')
        w_sector_upper_bound_slope = np.mean(np.diff(w_local_max_v) / np.diff(w_local_max_s_v))
        w_sector_lower_bound_slope = np.mean(np.diff(w_local_min_v) / np.diff(w_local_min_s_v))
        w_sector_middle_slope = np.mean([w_sector_upper_bound_slope, w_sector_lower_bound_slope])

        print(f'QuadraticCylindrical -- qp_initial:\n{qp_initial}')
        print(f'H_initial = {H_initial}')
        print(f'J_initial = {J_initial}')
        #print(f'scale_v   = {scale_v}')

        #scale       = np.mean(scale_v)
        #scale_error = np.max(scale_v) - np.min(scale_v)

        #print(f'scale_error = {scale_error}')
        #print(f'scale       = {scale}')

        # Take each segment and un-dilate it using an exponentially decreasing scale factor
        R_segment_v     = qp_interpolator(theta_local_max_s_v)[:,0,0]
        r_segment_v     = np.sqrt(R_segment_v)

        if len(theta_local_max_s_v) >= 2:
            # The correspondence between s (time parameter of dilating orbit) and t (time parameter
            # of non-dilating orbit) is
            #
            #     t = P*log_{lam}((lam-1)*s/(P*lam)),
            #
            # where lam is the scale factor and P is the period in the t time parameter, and log_{lam}
            # denotes the base-lam logarithm.

            # Pick arbitrary period for now, to be solved for later.
            #P = 1.0
            #P = theta_local_max_s_v[1] - theta_local_max_s_v[0] # This is semi-arbitrary
            theta_local_max_s_diff_v = np.diff(theta_local_max_s_v)
            lam_v = theta_local_max_s_diff_v[1:] / theta_local_max_s_diff_v[:-1]
            lam = np.mean(lam_v)
            lam_spread = np.max(lam_v) - np.min(lam_v)

            print(f'lam_v = {lam_v}')
            print(f'lam = {lam}')
            print(f'lam_spread = {lam_spread}')

            if lam < 1.0:
                # Collision is in the future
                s_collision = theta_local_max_s_v[0] + (theta_local_max_s_v[1] - theta_local_max_s_v[0])/(1.0-lam)
            elif lam > 1.0:
                # Collision is in the past -- TODO LATER
                s_collision = 0.0
            else: # lam == 1.0
                # No collision
                s_collision = 0.0

            # TODO: Actually solve for the correct function
            P = (theta_local_max_s_v[1] - theta_local_max_s_v[0]) / lam

            def tau (s:float) -> float:
                return P*np.log((lam-1.0)*(s-s_collision)/(P*lam))/np.log(lam)

            tau_theta_local_max_s_v = np.vectorize(tau)(theta_local_max_s_v)
            diff_tau_theta_local_max_s_v = np.diff(tau_theta_local_max_s_v)
            tau_diff_theta_local_max_spread = np.max(diff_tau_theta_local_max_s_v) - np.min(diff_tau_theta_local_max_s_v)
            print(f'tau_theta_local_max_s_v = {diff_tau_theta_local_max_s_v}')
            print(f'tau_diff_theta_local_max_spread = {tau_diff_theta_local_max_spread}')

        else:
            lam_v       = []
            lam         = np.nan
            lam_spread  = np.nan

            def tau (s:float) -> float:
                return s

        print(f'lam_v = {lam_v}')
        print(f'lam = {lam}')
        print(f'lam_spread = {lam_spread}')

        w_sector_middle_line = np.vectorize(lambda s:(s - s_collision) * w_sector_middle_slope)

        # TODO: Use Bezier interpolation instead
        scale_interpolator  = scipy.interpolate.interp1d(theta_local_max_s_v, r_segment_v)

        unwrapped_mask_v    = (t_v >= theta_local_max_s_v[0]) & (t_v <= theta_local_max_s_v[-1])
        unwrapped_s_v       = t_v[unwrapped_mask_v]
        unwrapped_t_v       = np.vectorize(tau)(unwrapped_s_v) + theta_local_max_s_v[0]
        unwrapped_qp_v      = np.copy(qp_v[unwrapped_mask_v])
        scale_v             = scale_interpolator(unwrapped_s_v)
        unwrapped_qp_v[:,0,0] /= scale_v**2
        unwrapped_qp_v[:,0,2] -= w_sector_middle_line(unwrapped_s_v)
        unwrapped_qp_v[:,0,2] /= scale_v**2
        unwrapped_qp_v[:,1,0] *= scale_v**2
        unwrapped_qp_v[:,1,2] *= scale_v**2

        H_unwrapped_qp_v = vorpy.apply_along_axes(vorpy.experimental.kh.QuadraticCylindricalNumerics.H__fast, (1,2), (unwrapped_qp_v,))
        J_unwrapped_qp_v = vorpy.apply_along_axes(vorpy.experimental.kh.QuadraticCylindricalNumerics.J__fast, (1,2), (unwrapped_qp_v,))

        print(f'min, max of H_unwrapped_qp_v: {np.min(H_unwrapped_qp_v), np.max(H_unwrapped_qp_v)}')
        print(f'min, max of J_unwrapped_qp_v: {np.min(J_unwrapped_qp_v), np.max(J_unwrapped_qp_v)}')

        euclidean_qp_v = vorpy.apply_along_axes(vorpy.experimental.kh.QuadraticCylindricalNumerics.qp_to_Euclidean__fast, (1,2), (qp_v,))

        euclidean_unwrapped_qp_v = vorpy.apply_along_axes(vorpy.experimental.kh.QuadraticCylindricalNumerics.qp_to_Euclidean__fast, (1,2), (unwrapped_qp_v,))

        for phase_index in range(2):
            s = 'p_' if phase_index == 1 else ''

            axis = axis_vv[phase_index][0]
            axis.set_title(f'initial ({s}x, {s}y) = {(euclidean_qp_v[0,phase_index,0], euclidean_qp_v[0,phase_index,1])}\n({s}x(t), {s}y(t))')
            axis.set_aspect(1.0)
            axis.plot(euclidean_qp_v[:,phase_index,0], euclidean_qp_v[:,phase_index,1])
            if p_R_initial != 0.0:
                axis.plot(euclidean_unwrapped_qp_v[:,phase_index,0], euclidean_unwrapped_qp_v[:,phase_index,1])

            axis = axis_vv[phase_index][1]
            axis.set_title(f'initial {s}R = {qp_v[0,phase_index,0]}\n(t, {s}R(t))')
            axis.plot(t_v, qp_v[:,phase_index,0])
            if p_R_initial != 0.0:
                axis.plot(unwrapped_t_v, unwrapped_qp_v[:,phase_index,0])
            if phase_index == 0:
                axis.axhline(0, color='black')
                # Draw the lines defining the sector that bounds the curve (t,R(t))
                s_vals = np.array(axis.get_xlim())
                R_vals = (s_vals - s_collision) * R_sector_lower_bound_slope
                axis.plot(s_vals, R_vals, color='red')
                R_vals = (s_vals - s_collision) * R_sector_middle_slope
                axis.plot(s_vals, R_vals, color='green')
                R_vals = (s_vals - s_collision) * R_sector_upper_bound_slope
                axis.plot(s_vals, R_vals, color='blue')

            axis = axis_vv[phase_index][2]
            axis.set_title(f'initial {s}theta = {qp_v[0,phase_index,1]}\n(t, {s}theta(t))')
            axis.plot(t_v, qp_v[:,phase_index,1])
            if p_R_initial != 0.0:
                axis.plot(unwrapped_t_v, unwrapped_qp_v[:,phase_index,1])

            axis = axis_vv[phase_index][3]
            axis.set_title(f'initial {s}w = {qp_v[0,phase_index,2]}\n(t, {s}w(t))')
            axis.plot(t_v, qp_v[:,phase_index,2])
            if p_R_initial != 0.0:
                axis.plot(unwrapped_t_v, unwrapped_qp_v[:,phase_index,2])
            if phase_index == 0:
                axis.axhline(0, color='black')
                # Draw the lines defining the sector that bounds the curve (t,w(t))
                s_vals = np.array(axis.get_xlim())
                R_vals = (s_vals - s_collision) * w_sector_lower_bound_slope
                axis.plot(s_vals, R_vals, color='red')
                R_vals = (s_vals - s_collision) * w_sector_middle_slope
                axis.plot(s_vals, R_vals, color='green')
                R_vals = (s_vals - s_collision) * w_sector_upper_bound_slope
                axis.plot(s_vals, R_vals, color='blue')

    #plot_p = base_dir_p / f'qp.p_y={p_y_initial}.png'
    plot_p.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
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

def transform_J (p_y_initial:float, other_trajectory_p_x_initial:float) -> None:
    base_dir_p = pathlib.Path(f'kh_dilation.transform_J')

    # TODO: Find a J = 0 orbit and a J != 0 orbit having the same delta in theta for each quasiperiod.
    # Is this the same as identical angular momentum (i.e. p_theta)?

    zero_J_trajectory_results = compute_trajectory(0.0, p_y_initial, base_dir_p)



    nonzero_J_trajectory_results = compute_trajectory(other_trajectory_p_x_initial, p_y_initial, base_dir_p)

    plot_trajectories([zero_J_trajectory_results, nonzero_J_trajectory_results], base_dir_p)

def unwrap_dilating_trajectory (p_R_initial:float, p_theta_initial:float, base_dir_p:pathlib.Path) -> None:
    pickle_file_p = base_dir_p / f'qp.p_R={p_R_initial}_p_theta={p_theta_initial}.pickle'
    pickle_file_p.parent.mkdir(parents=True, exist_ok=True)

    R_initial = 1.0
    theta_initial = 0.0
    w_initial = 0.0
    H_initial = 0.0
    solution_sheet = 0

    qp_initial = vorpy.experimental.kh.QuadraticCylindricalNumerics.qp_constrained_by_H__fast(
        np.array([R_initial, theta_initial, w_initial, p_R_initial, p_theta_initial, H_initial])
    )[solution_sheet]

    results = vorpy.experimental.kh.QuadraticCylindricalNumerics.compute_trajectory(
        pickle_file_p,
        qp_initial,
        t_final=100.0,
        solution_sheet=0,
    )

    plot_p = base_dir_p / f'qp.p_R={p_R_initial}_p_theta={p_theta_initial}.png'
    plot_trajectories_QuadraticCylindrical([results], plot_p)

if __name__ == '__main__':
    #for p_y_initial in np.linspace(0.05, 0.4, 20):
        #plot_J_equal_zero_extrapolated_trajectory(p_y_initial)

    #p_y_initial = 0.1
    #plot_J_equal_zero_extrapolated_trajectory(p_y_initial)

    #other_trajectory_p_x_initial = -0.1
    #for p_y_initial in np.linspace(0.05, 0.4, 20):
        #transform_J(p_y_initial, other_trajectory_p_x_initial)

    base_dir_p = pathlib.Path('unwrap.04')
    #for p_R_initial,p_theta_initial in itertools.product(np.linspace(-1.0/64, 1.0/64, 3), np.linspace(0.05, 0.4, 2)):
    for p_R_initial,p_theta_initial in itertools.product(np.linspace(-1.0/64, 0.0, 2, endpoint=False), np.linspace(0.05, 0.4, 2)):
        try:
            unwrap_dilating_trajectory(p_R_initial, p_theta_initial, base_dir_p)
        except ValueError as e:
            print(f'Caught {e}')
            pass
