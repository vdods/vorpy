import itertools
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import scipy.interpolate
import typing
import vorpy
import vorpy.experimental.bezier
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

def plot_trajectories (results_v:typing.Sequence[vorpy.integration.adaptive.IntegrateVectorFieldResults], base_dir_p:pathlib.Path, *, result_index=None) -> None:
    row_count   = 1
    col_count   = 2
    size        = 5
    fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

    def compute_collision_time (*, results:vorpy.integration.adaptive.IntegrateVectorFieldResults) -> float:
        # Don't bother if J is close to zero.
        J_initial = vorpy.experimental.kh.EuclideanNumerics.J__fast(results.y_t[0])
        if np.abs(J_initial) < 1.0e-6:
            return np.nan

        t_v = results.t_v
        z_v = results.y_t[:,0,2]
        _,_,z_zero_pos_v = vorpy.realfunction.piecewiselinear.oriented_zeros(z_v, t_v=t_v, orientation_p=(lambda o:o>0))

        print(f'J_initial = {J_initial}, z_zero_pos_v = {z_zero_pos_v}')

        if len(z_zero_pos_v) < 3:
            print(f'exiting because len(z_zero_pos_v) (which was {len(z_zero_pos_v)}) was < 3')
            return np.nan

        z_zero_pos_delta_v = np.diff(z_zero_pos_v)
        lam_v = z_zero_pos_delta_v[1:] / z_zero_pos_delta_v[:-1]
        lam = np.mean(lam_v)

        # Too close to non-dilating to have a collision
        if np.abs(lam - 1.0) < 1.0e-3:
            print(f'exiting because lam (which was {lam}) was too close to 1')
            return np.nan

        collision_time = z_zero_pos_v[0] + z_zero_pos_delta_v[0] / (1.0 - lam)
        print(f'lam = {lam}, lam_v stddev: {np.std(lam_v)}, collision_time = {collision_time}')
        return collision_time

    for results in results_v:
        t_v = results.t_v
        qp_v = results.y_t

        p_x_initial = qp_v[0,1,0]
        p_y_initial = qp_v[0,1,1]

        collision_time = compute_collision_time(results=results)
        if np.isfinite(collision_time):
            # Assume that J is a "rate" of size / time.  Then with the fact that J is conserved
            # (assuming that H = 0), then the size must be changing linearly with time.
            J_initial = vorpy.experimental.kh.EuclideanNumerics.J__fast(results.y_t[0])

            precollision_mask_v = t_v < collision_time
            precollision_t_v = t_v[precollision_mask_v]
            if len(precollision_t_v) > 0:
                size_v = -J_initial * (collision_time - precollision_t_v)
                print(f'size_v[0] = {size_v[0]}')

                normalized_size_v = (collision_time - precollision_t_v) / (collision_time - precollision_t_v[0])

                #axis = axis_vv[1][3]
                #axis.set_title('"size"')
                #axis.axhline(0.0, color='black')
                #axis.plot(precollision_t_v, size_v)
                ##axis.plot(precollision_t_v, normalized_size_v)
                #axis.axvline(collision_time)
            else:
                #axis = axis_vv[1][3]
                #axis.set_title('"size" could not be computed')
                normalized_size_v = None

        def dilation (lam:float, qp:np.ndarray) -> np.ndarray:
            assert qp.shape == (2,3)
            retval = np.copy(qp)
            retval[0,0:2] *= lam
            retval[0,2]   *= lam**2
            retval[1,0:2] /= lam
            retval[1,2]   /= lam**2
            return retval

        if np.isfinite(collision_time) and normalized_size_v is not None:
            # Apply a progressive dilation (using sqrt(lam) because lam is based on z)
            precollision_qp_v = qp_v[precollision_mask_v,...]
            dilated_precollision_qp_v = np.copy(precollision_qp_v)
            for i,normalized_size in enumerate(normalized_size_v):
                dilated_precollision_qp_v[i] = dilation(1.0 / np.sqrt(normalized_size), dilated_precollision_qp_v[i])
        else:
            dilated_precollision_qp_v = None

        for phase_index in range(1):
            s = 'p_' if phase_index == 1 else ''

            axis = axis_vv[phase_index][0]
            #axis.set_title(f'initial ({s}x, {s}y) = {(qp_v[0,phase_index,0], qp_v[0,phase_index,1])}\n({s}x(t), {s}y(t))')
            axis.set_aspect(1.0)
            if dilated_precollision_qp_v is not None:
                axis.plot(dilated_precollision_qp_v[:,phase_index,0], dilated_precollision_qp_v[:,phase_index,1])
            else:
                axis.plot(qp_v[:,phase_index,0], qp_v[:,phase_index,1])
            axis.plot([0.0], [0.0], '.', color='black')

            for coord_index,coord_name in enumerate(['z'], 2):
                axis = axis_vv[phase_index][1+coord_index-2]
                #axis.set_title(f'initial {s}{coord_name} = {qp_v[0,phase_index,coord_index]}\n(t, {s}{coord_name}(t))')
                if dilated_precollision_qp_v is not None:
                    axis.plot(t_v, dilated_precollision_qp_v[:,phase_index,coord_index])
                else:
                    axis.plot(t_v, qp_v[:,phase_index,coord_index])

        #p_theta_v   = vorpy.apply_along_axes(vorpy.experimental.kh.EuclideanNumerics.p_theta__fast, (1,2), (results.y_t,))
        #H_v         = vorpy.apply_along_axes(vorpy.experimental.kh.EuclideanNumerics.H__fast, (1,2), (results.y_t,))
        #J_v         = vorpy.apply_along_axes(vorpy.experimental.kh.EuclideanNumerics.J__fast, (1,2), (results.y_t,))

        #axis = axis_vv[0][4]
        #axis.set_title(f'p_theta = {p_theta_v[0]}')
        #axis.plot(t_v, p_theta_v)

        #axis = axis_vv[1][4]
        #axis.set_title(f'H = {H_v[0]}')
        #axis.plot(t_v, H_v)

        #axis = axis_vv[1][5]
        #axis.set_title(f'J = {J_v[0]}')
        #axis.plot(t_v, J_v)

    if len(results_v) == 1:
        plot_p = base_dir_p / f'qp.result_index={result_index}.x={results.y_t[0,0,0]}.y={results.y_t[0,0,1]}.z={results.y_t[0,0,2]}.p_x={results.y_t[0,1,0]}.p_y={results.y_t[0,1,1]}.p_z={results.y_t[0,1,2]}.pdf'
    else:
        plot_p = base_dir_p / f'results.pdf'

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

def plot_z_axis_trajectories (*, z_initial_v:typing.List[float], p_z_initial_v:typing.List[float], plot_dir_p:pathlib.Path):
    solution_sheet = 0

    x_initial = 0.0
    y_initial = 0.0
    p_y_initial = 0.0

    plot_dir_p.mkdir(parents=True, exist_ok=True)
    t_final = 2000.0
    results_v = []
    for z_initial in z_initial_v:
        p_x_initial = 0.25 / np.sqrt(np.pi*z_initial)
        for p_z_initial in p_z_initial_v:
            qp_initial = np.array([
                [x_initial,   y_initial,   z_initial],
                [p_x_initial, p_y_initial, p_z_initial]
            ])

            results = vorpy.experimental.kh.EuclideanNumerics.compute_trajectory(
                #plot_dir_p / f'p_theta={p_theta_initial}.J={J_initial}.pickle',
                plot_dir_p / f'z={z_initial}.p_z={p_z_initial}.pickle',
                qp_initial,
                t_final=t_final,
                solution_sheet=solution_sheet, # This isn't used in the computation, it's just stored in the pickle file
                return_y_jet=False,
            )
            plot_trajectories(results_v=[results], base_dir_p=plot_dir_p, result_index=len(results_v))
            results_v.append(results)

    plot_trajectories(results_v=results_v, base_dir_p=plot_dir_p)

if __name__ == '__main__':
    z_initial_v = [1.0]
    #p_z_initial_v = [0.0]+[0.5**p for p in np.linspace(0.0, 5.0, num=33)]
    #p_z_initial_v = np.linspace(0.127, 0.158, num=16)
    p_z_initial_v = np.linspace(0.0, 0.3, num=10)
    plot_z_axis_trajectories(z_initial_v=z_initial_v, p_z_initial_v=p_z_initial_v, plot_dir_p=pathlib.Path('kh_dilation.z-axis.11'))
