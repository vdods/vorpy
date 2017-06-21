import numpy as np
import vorpy

def deviation_form (*, t_v, qp_v, dH_dq, dH_dp):
    """
    Hamilton's equations are

        dq/dt =   \partial H / \partial p
        dp/dt = - \partial H / \partial q

    This function computes the deviation of the discretized curve given by time values t_v and coordinates
    qp_v from satisfaction of Hamilton's equations.  A perfect solution would have a deviation of zero at all
    time/point pairs (though because we're dealing with a discretized curve, this wouldn't imply that the
    solution is exact).

    The value returned is the evaluation of a form equivalent to Hamilton's equations:

        dq/dt - \partial H / \partial p
        dp/dt + \partial H / \partial q

    and has the shape (T,A_1,A_2,...,A_K,2,N), which will be the same shape as qp_v, noting that K may be 0.
    Note that because t_v indexes the time values corresponding to the elements of qp_v.

    The [discrete] derivatives dq/dt and dp/dt are computed using a symmetric difference quotient:

        (dq/dt)[i] := (q[i+1] - q[i-1]) / (t[i+1] - t[i-1])

    with (dp/dt)[i] defined analogously.  For the first and last entries, a non-symmetric difference quotient
    will be used.

        (dq/dt)[ 0] := (q[ 1] - q[ 0]) / (t[ 1] - t[ 0])
        (dq/dt)[-1] := (q[-1] - q[-2]) / (t[-1] - t[-2])

    with (dp/dt)[0] and (dp/dt)[-1] defined analogously.

    TODO: Maybe only compute this for interior samples; end-point difference quotients might not be that useful.
    """

    #print('deviation_form; qp_v.shape = {0}'.format(qp_v.shape))

    assert np.all(np.diff(t_v) > 0), 't_v must be an increasing sequence of time values' # TODO: allow strictly decreasing time values
    assert t_v.shape[0] == qp_v.shape[0]
    assert t_v.shape[0] >= 2, 'no discrete derivative can be defined without at least two time values'
    assert qp_v.shape[-2] == 2

    # T is the number of time values
    T = t_v.shape[0]
    # N is the dimension of configuration space.
    N = qp_v.shape[-1]

    retval = np.ndarray(qp_v.shape, dtype=qp_v.dtype)
    # First populate retval with discrete derivatives of q and p.
    #print('deviation_form; retval[0,...].shape = {0}'.format(retval[0,...].shape))
    #print('deviation_form; retval[1:-1,...].shape = {0}'.format(retval[1:-1,...].shape))
    #print('deviation_form; (qp_v[2:] - qp_v[:-2]).shape = {0}'.format((qp_v[2:] - qp_v[:-2]).shape))
    #print('deviation_form; (t_v[2:] - t_v[:-2]).shape = {0}'.format((t_v[2:] - t_v[:-2]).shape))
    retval[0,...] = (qp_v[1] - qp_v[0]) / (t_v[1] - t_v[0])
    retval[1:-1,...] = qp_v[2:] - qp_v[:-2]
    retval[1:-1,...] /= (t_v[2:] - t_v[:-2]).reshape((T-2,)+(1,)*(len(retval.shape)-1)) # The reshaping is necessary for correct broadcasting
    retval[-1,...] = (qp_v[-1] - qp_v[-2]) / (t_v[-1] - t_v[-2])
    # Then subtract off the partials of the Hamiltonian in order to get the deviation form
    retval[...,0,:] -= vorpy.apply_along_axes(dH_dp, (-1,), (qp_v[...,0,:], qp_v[...,1,:]), output_axis_v=(-1,), func_output_shape=(N,))
    retval[...,1,:] += vorpy.apply_along_axes(dH_dq, (-1,), (qp_v[...,0,:], qp_v[...,1,:]), output_axis_v=(-1,), func_output_shape=(N,))

    return retval
