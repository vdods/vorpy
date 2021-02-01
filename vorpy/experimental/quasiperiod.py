import matplotlib.pyplot as plt
import numpy as np
import pathlib
import scipy.interpolate
import sys
import typing
import vorpy.pickle

def zero_crossing_times (t_v:np.ndarray, f_v:np.ndarray, *, orientation:int=0) -> np.ndarray:
    """
    Returns piecewise-linearly-computed approximations of the actual zero crossing times of the function (t,f(t))
    (where the function defined on the elements of t_v by t_v[i] |-> f_v[i]), and the pairs of indices [of t_v]
    that the zero crossings occur between.

    The orientation parameter may be used to specify the orientation of zero crossings to return.
    -   orientation == 0 : return all zero crossings
    -   orientation < 0  : return negatively oriented zero crossings (where the function goes from positive to negative)
    -   orientation > 0  : return positively oriented zero crossings (where the function goes from negative to positive)
    """

    if len(t_v) != len(f_v):
        raise TypeError(f'expected len(t_v) == len(f_v), but got len(t_v) = {len(t_v)} and len(f_v) = {len(f_v)}')

    # zc stands for zero crossing.

    # Non-positive elements of this indicate a zero crossing.
    zc_discriminant_v = f_v[:-1] * f_v[1:]
    # Consider only strictly negative discriminant as indicating a zero crossing.  This will not pick up
    # cases where there is a repeated zero, or where the function touches but doesn't cross zero.
    zc_v = zc_discriminant_v < 0
    zc_index_v = np.where(zc_v)[0]
    assert np.all(zc_index_v < len(t_v)-1)
    if orientation != 0:
        zc_orientation_v = np.sign(f_v[zc_index_v+1] - f_v[zc_index_v])
        assert np.all(zc_orientation_v != 0), 'this should be true by construction (following the zc_discriminant_v < 0 condition)'
        zc_index_v = zc_index_v[zc_orientation_v == np.sign(orientation)]

    assert np.all(np.sign(f_v[zc_index_v+1]) != np.sign(f_v[zc_index_v]))
    assert np.all(f_v[zc_index_v+1]*f_v[zc_index_v] < 0), 'this should be equivalent to the sign check, but is done using discriminant'
    if orientation != 0:
        assert np.all(np.sign(f_v[zc_index_v+1]) == np.sign(orientation))

    zc_index_pair_t = np.ndarray((len(zc_index_v),2), dtype=int)
    zc_index_pair_t[:,0] = zc_index_v
    zc_index_pair_t[:,1] = zc_index_v+1
    assert np.all(zc_index_pair_t < len(t_v)), 'each element of zc_index_pair_t should be a valid index for both t_v and f_v'

    # Make tensors quantifying the intervals containing the zero crossings.
    # Note here that because zc_index_pair_t is a 2-tensor, and t_v and f_v are 1-tensors,
    # zc_interval_t_v and zc_interval_f_v will be a 2-tensor whose rows are the interval bounds.
    zc_interval_t_v = t_v[zc_index_pair_t]
    zc_interval_f_v = f_v[zc_index_pair_t]
    assert zc_interval_t_v.shape == (len(zc_index_v),2)
    assert zc_interval_f_v.shape == (len(zc_index_v),2)

    # For each zero crossing, use a piecewise linear interpolation of f_v to solve for a better
    # approximation of the exact time it crosses zero.
    zc_t_delta_v = np.diff(zc_interval_t_v, axis=1).reshape(-1)
    zc_f_delta_v = np.diff(zc_interval_f_v, axis=1).reshape(-1)
    zc_t_v = zc_interval_t_v[:,0] - zc_interval_f_v[:,0]*zc_t_delta_v/zc_f_delta_v

    ## Numerical sanity check (the bound is based on the max number encountered in the solution for the respective component of zc_t_v).
    #assert np.all(np.interp(zc_t_v, t_v, f_v) < 1.0e-8*np.max(zc_interval_f_v, axis=1))

    return zc_t_v, zc_index_pair_t

def critical_points (t_v:np.ndarray, f_v:np.ndarray, *, orientation:int=0) -> np.ndarray:
    """
    Returns a tensor C of shape (k,2), where the ith critical point (t_i,f_i) is (C[i,0], C[i,1]), and the pairs of
    indices [of t_v] that the critical points occur between.
    """

    if len(t_v) != len(f_v):
        raise TypeError(f'expected len(t_v) == len(f_v), but got len(t_v) = {len(t_v)} and len(f_v) = {len(f_v)}')

    # Use a symmetric definition of derivative.
    discrete_deriv_f_v = (f_v[2:] - f_v[:-2]) / (t_v[2:] - t_v[:-2])
    critical_point_t_v, critical_point_index_pair_t = zero_crossing_times(t_v[1:-1], discrete_deriv_f_v, orientation=orientation)
    critical_point_t = np.ndarray((len(critical_point_t_v),2), dtype=critical_point_t_v.dtype)
    critical_point_t[:,0] = critical_point_t_v
    critical_point_t[:,1] = np.interp(critical_point_t_v, t_v, f_v)
    return critical_point_t, critical_point_index_pair_t

def local_maxima (t_v:np.ndarray, f_v:np.ndarray) -> np.ndarray:
    return critical_points(t_v, f_v, orientation=-1)

def local_minima (t_v:np.ndarray, f_v:np.ndarray) -> np.ndarray:
    return critical_points(t_v, f_v, orientation=1)

def compute_lambda_v (x_v:np.ndarray, *, name_o:typing.Optional[str]=None) -> np.ndarray:
    prefix = '' if name_o is None else f'{name_o} '
    pos_diff_v = np.diff(x_v)
    lambda_v = pos_diff_v[1:] / pos_diff_v[:-1]
    print(f'{prefix}lambda_v = {lambda_v}')
    if len(lambda_v) > 0:
        lambda_range = (np.min(lambda_v), np.max(lambda_v))
        lambda_range_size = lambda_range[1] - lambda_range[0]
    else:
        lambda_range = (np.nan, np.nan)
        lambda_range_size = np.nan
    print(f'{prefix}lambda_v in range {lambda_range}')
    print(f'{prefix}lambda_v range size = {lambda_range_size}')
    return lambda_v, lambda_range, lambda_range_size

def do_stuff (*, pickle_p:pathlib.Path, plot_p:pathlib.Path, plot_momentum=False) -> None:
    data_d = vorpy.pickle.unpickle(pickle_filename=pickle_p, log_out=sys.stdout)

    results = data_d['results']

    t_v = results.t_v

    R_v = results.y_t[:,0,0]
    p_R_v = results.y_t[:,1,0]

    R_local_maximum_t, _ = local_maxima(t_v, R_v)

    R_lambda_v, R_lambda_range, R_lambda_range_size = compute_lambda_v(R_local_maximum_t[:,0], name_o='R')
    #R_quasiperiod = R_local_maximum_t[1,0] - R_local_maximum_t[0,0]

    theta_v = results.y_t[:,0,1]
    p_theta_v = results.y_t[:,1,1]

    #theta_critical_point_t, _ = critical_points(t_v, theta_v)
    theta_local_maximum_t, _ = local_maxima(t_v, theta_v)
    theta_local_minimum_t, _ = local_minima(t_v, theta_v)

    theta_lambda_v, theta_lambda_range, theta_lambda_range_size = compute_lambda_v(theta_local_maximum_t[:,0], name_o='theta')
    #theta_quasiperiod = theta_local_maximum_t[1,0] - theta_local_maximum_t[0,0]

    w_v = results.y_t[:,0,2]
    w_zero_crossing_v, _ = zero_crossing_times(t_v, w_v)
    w_zero_crossing_pos_v, _ = zero_crossing_times(t_v, w_v, orientation=1)
    w_zero_crossing_neg_v, _ = zero_crossing_times(t_v, w_v, orientation=-1)

    w_lambda_v, w_lambda_range, w_lambda_range_size = compute_lambda_v(w_zero_crossing_pos_v, name_o='w')

    p_w_v           = results.y_t[:,1,2]

    rho_v           = np.sqrt(R_v**2 + w_v**2)

    J_v             = 2*(R_v*p_R_v + w_v*p_w_v)
    J_initial       = J_v[0]
    J_mean          = np.mean(J_v)

    sqrt_R_initial  = np.sqrt(R_v[0])
    P_R_initial     = 2.0*sqrt_R_initial*p_R_v[0]
    P_theta_initial = p_theta_v[0]/sqrt_R_initial + 2.0*sqrt_R_initial*p_w_v[0]
    H_initial       = (P_R_initial**2 + P_theta_initial**2)/2 - 1.0/(8.0*np.pi*np.sqrt(R_v[0]**2 + w_v[0]**2))

    # Collate lambda values
    lambda_v = []
    if R_lambda_range_size < 1.0e-4:
        lambda_v.extend(R_lambda_v)
    if theta_lambda_range_size < 1.0e-4:
        lambda_v.extend(theta_lambda_v)
    if w_lambda_range_size < 1.0e-4:
        lambda_v.extend(w_lambda_v)
    lambda_v = np.array(lambda_v)

    if len(lambda_v) > 0:
        lambda_range = (np.min(lambda_v), np.max(lambda_v))
        lambda_range_size = np.diff(lambda_range)
    else:
        lambda_range = (np.nan, np.nan)
        lambda_range_size = np.nan

    if np.isfinite(lambda_range_size) and lambda_range_size < 1.0e-4 and len(w_zero_crossing_pos_v) >= 2:
        lam                     = np.mean(lambda_v)

        ## Record lambda vs J_initial
        #vorpy.pickle.pickle(
            #data=dict(
                #coordinates_name='QuadraticCylindrical',
                #qp_initial=results.y_t[0],
                #lam=lam,
                #J_initial=J_initial,
            #),
            #pickle_filename=str(pickle_p)+'.J_vs_lam.pickle',
            #log_out=sys.stdout,
        #)

        quasiperiod_t_range     = (w_zero_crossing_pos_v[0], w_zero_crossing_pos_v[1])
        quasiperiod             = np.diff(quasiperiod_t_range)[0]
        theta_delta             = theta_local_maximum_t[1,1] - theta_local_maximum_t[0,1]

        y_t_interpolator        = scipy.interpolate.interp1d(t_v, results.y_t, axis=0)

        extrapolated_t_v        = np.linspace(quasiperiod_t_range[0], quasiperiod_t_range[1], 10000)
        extrapolated_y_t        = y_t_interpolator(extrapolated_t_v)

        extrapolated_R_v        = extrapolated_y_t[:,0,0]
        extrapolated_theta_v    = extrapolated_y_t[:,0,1]
        extrapolated_w_v        = extrapolated_y_t[:,0,2]

        extrapolated_p_R_v      = extrapolated_y_t[:,1,0]
        extrapolated_p_theta_v  = extrapolated_y_t[:,1,1]
        extrapolated_p_w_v      = extrapolated_y_t[:,1,2]

        # Transform the extrapolated curve
        extrapolated_t_v       -= quasiperiod_t_range[0]
        extrapolated_t_v       *= lam
        extrapolated_t_v       += quasiperiod_t_range[1]

        #extrapolated_R_v[:]     = 0.5*np.log(lam*np.exp(2.0*extrapolated_R_v))
        extrapolated_R_v       *= lam
        extrapolated_theta_v   += theta_delta
        extrapolated_w_v       *= lam

        extrapolated_p_R_v     /= lam
        extrapolated_p_w_v     /= lam

        # TODO: extrapolate momentum

        # Sample the actual solution curve at the extrapolated time values and compare.
        valid_t_mask_v          = extrapolated_t_v <= t_v[-1]
        valid_t_v               = extrapolated_t_v[valid_t_mask_v]
        sampled_y_t             = y_t_interpolator(valid_t_v)
        extrapolation_error_v   = np.max(np.abs(sampled_y_t[valid_t_mask_v,:,:] - extrapolated_y_t[valid_t_mask_v,:,:]), axis=0)
        extrapolation_error     = np.max(extrapolation_error_v)
        print(f'\n\nextrapolation_error_v = {extrapolation_error_v}\n\n')
        print(f'\n\nextrapolation_error = {extrapolation_error}\n\n')
    else:
        print('NO UNIQUE LAMBDA, SKIPPING QUASI-PERIODIC SOLUTION SOLVE')
        lam                     = None
        extrapolated_t_v        = None
        extrapolated_y_t        = None
        extrapolation_error     = None

    row_count   = 2 if plot_momentum else 1
    col_count   = 2
    #size        = 8
    size        = 5
    fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

    qp_t = np.ndarray((len(t_v),2,3), dtype=float)

    # Change coordinates back into Cartesian
    sqrt_R_v    = np.sqrt(R_v)

    qp_t[:,0,0] = sqrt_R_v*np.cos(theta_v)
    qp_t[:,0,1] = sqrt_R_v*np.sin(theta_v)
    qp_t[:,0,2] = w_v

    qp_t[:,1,0] = 2*sqrt_R_v*p_R_v*np.cos(theta_v) - p_theta_v*np.sin(theta_v)/sqrt_R_v
    qp_t[:,1,1] = 2*sqrt_R_v*p_R_v*np.sin(theta_v) + p_theta_v*np.cos(theta_v)/sqrt_R_v
    qp_t[:,1,2] = p_w_v

    # Sanity check
    euclidean_J_v = qp_t[:,0,0]*qp_t[:,1,0] + qp_t[:,0,1]*qp_t[:,1,1] + 2*qp_t[:,0,2]*qp_t[:,1,2]
    J_v_error = np.max(np.abs(euclidean_J_v - J_v))
    print(f'J_v_error = {J_v_error}')

    qp_t_interpolator = scipy.interpolate.interp1d(t_v, qp_t, axis=0)

    if extrapolated_y_t is not None:
        extrapolated_qp_t = np.ndarray((len(extrapolated_t_v),2,3), dtype=float)

        # Change coordinates back into cylindrical
        extrapolated_R_v = extrapolated_y_t[:,0,0]
        extrapolated_p_R_v = extrapolated_y_t[:,1,0]

        extrapolated_theta_v = extrapolated_y_t[:,0,1]
        extrapolated_p_theta_v = extrapolated_y_t[:,1,1]

        extrapolated_w_v = extrapolated_y_t[:,0,2]
        extrapolated_p_w_v = extrapolated_y_t[:,1,2]

        # Change coordinates back into Cartesian
        sqrt_extrapolated_R_v = np.sqrt(extrapolated_R_v)

        extrapolated_qp_t[:,0,0] = sqrt_extrapolated_R_v*np.cos(extrapolated_theta_v)
        extrapolated_qp_t[:,0,1] = sqrt_extrapolated_R_v*np.sin(extrapolated_theta_v)
        extrapolated_qp_t[:,0,2] = extrapolated_w_v

        extrapolated_qp_t[:,1,0] = 2*sqrt_extrapolated_R_v*extrapolated_p_R_v*np.cos(extrapolated_theta_v) - extrapolated_p_theta_v*np.sin(extrapolated_theta_v)/sqrt_extrapolated_R_v
        extrapolated_qp_t[:,1,1] = 2*sqrt_extrapolated_R_v*extrapolated_p_R_v*np.sin(extrapolated_theta_v) + extrapolated_p_theta_v*np.cos(extrapolated_theta_v)/sqrt_extrapolated_R_v
        extrapolated_qp_t[:,1,2] = extrapolated_p_w_v

        source_t_mask_v = (quasiperiod_t_range[0] <= t_v) & (t_v <= quasiperiod_t_range[1])
    else:
        extrapolated_qp_t = None

    axis = axis_vv[0][0]
    #axis.set_title(f'Plot of (x(t),y(t))\nInitial conditions (x,y,z,p_x,p_y,p_z):\n{tuple(qp_t[0,:,:].reshape(-1).tolist())}\nPurple segment: source fundamental domain\nOrange segment: extrapolated fundamental domain')
    axis.set_aspect(1.0)
    axis.plot([0], [0], '.', color='black')
    axis.plot(qp_t[:,0,0], qp_t[:,0,1])
    if extrapolated_qp_t is not None:
        axis.plot(qp_t[source_t_mask_v,0,0], qp_t[source_t_mask_v,0,1], color='purple')
        axis.plot(extrapolated_qp_t[:,0,0], extrapolated_qp_t[:,0,1], color='orange')

    # Make the plot square
    axis_xlim_old = axis.get_xlim()
    axis_ylim_old = axis.get_ylim()
    axis_x_size = abs(axis_xlim_old[1] - axis_xlim_old[0])
    axis_y_size = abs(axis_ylim_old[1] - axis_ylim_old[0])
    axis_size = max(axis_x_size, axis_y_size)
    if axis_x_size < axis_size:
        difference = axis_size - axis_x_size
        axis.set_xlim(axis_xlim_old[0]-difference/2.0, axis_xlim_old[1]+difference/2.0)
    if axis_y_size < axis_size:
        difference = axis_size - axis_y_size
        axis.set_ylim(axis_ylim_old[0]-difference/2.0, axis_ylim_old[1]+difference/2.0)

    axis = axis_vv[1][0]
    #axis.set_title(f'(p_x(t),p_y(t))\npurple: source fund. domain\norange: extrap\'ed fund. domain')
    axis.set_aspect(1.0)
    axis.plot([0], [0], '.', color='black')
    axis.plot(qp_t[:,1,0], qp_t[:,1,1])
    if extrapolated_qp_t is not None:
        axis.plot(qp_t[source_t_mask_v,1,0], qp_t[source_t_mask_v,1,1], color='purple')
        axis.plot(extrapolated_qp_t[:,1,0], extrapolated_qp_t[:,1,1], color='orange')

    # Make the plot square
    axis_xlim_old = axis.get_xlim()
    axis_ylim_old = axis.get_ylim()
    axis_x_size = abs(axis_xlim_old[1] - axis_xlim_old[0])
    axis_y_size = abs(axis_ylim_old[1] - axis_ylim_old[0])
    axis_size = max(axis_x_size, axis_y_size)
    if axis_x_size < axis_size:
        difference = axis_size - axis_x_size
        axis.set_xlim(axis_xlim_old[0]-difference/2.0, axis_xlim_old[1]+difference/2.0)
    if axis_y_size < axis_size:
        difference = axis_size - axis_y_size
        axis.set_ylim(axis_ylim_old[0]-difference/2.0, axis_ylim_old[1]+difference/2.0)

    #axis = axis_vv[0][1]
    #axis.set_title(f'(t,R(t))\npurple: source fund. domain\norange: extrap\'ed fund. domain')
    #axis.axhline(0, color='black')
    #axis.plot(t_v, R_v)
    #if extrapolated_y_t is not None:
        #axis.plot(t_v[source_t_mask_v], R_v[source_t_mask_v], color='purple')
        #axis.plot(extrapolated_t_v, extrapolated_y_t[:,0,0], color='orange')
        #axis.axvline(quasiperiod_t_range[0], color='black', alpha=0.5)
        #axis.axvline(quasiperiod_t_range[1], color='black', alpha=0.5)
    #for R_local_maximum in R_local_maximum_t:
        #axis.axvline(R_local_maximum[0], color='green', alpha=0.3)
        #axis.axhline(R_local_maximum[1], color='green', alpha=0.3)

    #axis = axis_vv[1][1]
    #axis.set_title(f'(t,p_R(t)) (R = log(r))\npurple: source fund. domain\norange: extrap\'ed fund. domain')
    #axis.axhline(0, color='black')
    #axis.plot(t_v, p_R_v)
    #if extrapolated_y_t is not None:
        #axis.plot(t_v[source_t_mask_v], p_R_v[source_t_mask_v], color='purple')
        #axis.plot(extrapolated_t_v, extrapolated_y_t[:,1,0], color='orange')
        #axis.axvline(quasiperiod_t_range[0], color='black', alpha=0.5)
        #axis.axvline(quasiperiod_t_range[1], color='black', alpha=0.5)

    #axis = axis_vv[0][2]
    #axis.set_title(f'(t,theta(t))\ntheta_lambda range = {theta_lambda_range}\ntheta_lambda range size = {theta_lambda_range_size}')
    #axis.axhline(0, color='black')
    #axis.plot(t_v, theta_v)
    #if extrapolated_y_t is not None:
        #axis.plot(t_v[source_t_mask_v], theta_v[source_t_mask_v], color='purple')
        #axis.plot(extrapolated_t_v, extrapolated_y_t[:,0,1], color='orange')
        #axis.axvline(quasiperiod_t_range[0], color='black', alpha=0.5)
        #axis.axvline(quasiperiod_t_range[1], color='black', alpha=0.5)
    #for theta_local_maximum in theta_local_maximum_t:
        #axis.axvline(theta_local_maximum[0], color='green', alpha=0.3)
        #axis.axhline(theta_local_maximum[1], color='green', alpha=0.3)

    #axis = axis_vv[1][2]
    #axis.set_title(f'(t,p_theta(t))\nlambda used for extrapolation = {lam}\nextrapolation error = {extrapolation_error}')
    #axis.axhline(0, color='black')
    #axis.plot(t_v, p_theta_v)
    #if extrapolated_y_t is not None:
        #axis.plot(t_v[source_t_mask_v], p_theta_v[source_t_mask_v], color='purple')
        #axis.plot(extrapolated_t_v, extrapolated_y_t[:,1,1], color='orange')
        #axis.axvline(quasiperiod_t_range[0], color='black', alpha=0.5)
        #axis.axvline(quasiperiod_t_range[1], color='black', alpha=0.5)

    #axis = axis_vv[0][3]
    axis = axis_vv[0][1]
    #axis.set_title(f'(t,w(t))\nw_lambda range = {w_lambda_range}\nw_lambda range size = {w_lambda_range_size}')
    #axis.set_title(f'Plot of (t,z(t))\nH = {H_initial}, J = {J_initial}\nlambda = {lam}')
    axis.axhline(0, color='black')
    #axis.plot(t_v, w_v)
    axis.plot(t_v, w_v/4) # w = 4*z
    if extrapolated_y_t is not None:
        #axis.plot(t_v[source_t_mask_v], w_v[source_t_mask_v], color='purple')
        axis.plot(t_v[source_t_mask_v], w_v[source_t_mask_v]/4, color='purple')
        #axis.plot(extrapolated_t_v, extrapolated_y_t[:,0,2], color='orange')
        axis.plot(extrapolated_t_v, extrapolated_y_t[:,0,2]/4, color='orange')
        axis.axvline(quasiperiod_t_range[0], color='black', alpha=0.5)
        axis.axvline(quasiperiod_t_range[1], color='black', alpha=0.5)
    #for w_zero_crossing in w_zero_crossing_v:
        #axis.axvline(w_zero_crossing, color='black', alpha=0.3)
    for w_zero_crossing_pos in w_zero_crossing_pos_v:
        axis.axvline(w_zero_crossing_pos, color='green', alpha=0.3)
    for w_zero_crossing_neg in w_zero_crossing_neg_v:
        axis.axvline(w_zero_crossing_neg, color='red', alpha=0.3)

    axis = axis_vv[1][1]
    #axis.set_title(f'(t,p_w(t))\nw_lambda range = {w_lambda_range}\nw_lambda range size = {w_lambda_range_size}')
    #axis.set_title(f'Plot of (t,p_z(t))\nH = {H_initial}, J = {J_initial}\nlambda = {lam}')
    axis.axhline(0, color='black')
    #axis.plot(t_v, w_v)
    axis.plot(t_v, p_w_v*4) # w = 4*z, so p_w = z/4
    if extrapolated_y_t is not None:
        #axis.plot(t_v[source_t_mask_v], w_v[source_t_mask_v], color='purple')
        axis.plot(t_v[source_t_mask_v], p_w_v[source_t_mask_v]*4, color='purple')
        #axis.plot(extrapolated_t_v, extrapolated_y_t[:,0,2], color='orange')
        axis.plot(extrapolated_t_v, extrapolated_y_t[:,1,2]*4, color='orange')
        axis.axvline(quasiperiod_t_range[0], color='black', alpha=0.5)
        axis.axvline(quasiperiod_t_range[1], color='black', alpha=0.5)
    #for w_zero_crossing in w_zero_crossing_v:
        #axis.axvline(w_zero_crossing, color='black', alpha=0.3)
    for w_zero_crossing_pos in w_zero_crossing_pos_v:
        axis.axvline(w_zero_crossing_pos, color='green', alpha=0.3)
    for w_zero_crossing_neg in w_zero_crossing_neg_v:
        axis.axvline(w_zero_crossing_neg, color='red', alpha=0.3)

    #axis = axis_vv[1][3]
    #axis.set_title(f'(t,p_w(t))\nJ_initial = {J_initial}')
    #axis.axhline(0, color='black')
    #axis.plot(t_v, p_w_v)
    #if extrapolated_y_t is not None:
        #axis.plot(t_v[source_t_mask_v], p_w_v[source_t_mask_v], color='purple')
        #axis.plot(extrapolated_t_v, extrapolated_y_t[:,1,2], color='orange')
        #axis.axvline(quasiperiod_t_range[0], color='black', alpha=0.5)
        #axis.axvline(quasiperiod_t_range[1], color='black', alpha=0.5)

    #axis = axis_vv[0][4]
    #axis.set_title(f'(t,rho(t))\nrho = sqrt(R^2 + w^2)\nH_initial = {H_initial}, J_initial = {J_initial}')
    #axis.axhline(0, color='black')
    #axis.plot(t_v, rho_v)

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
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <pickle> <plot.png>')
        sys.exit(-1)

    do_stuff(pickle_p=pathlib.Path(sys.argv[1]), plot_p=pathlib.Path(sys.argv[2]))

