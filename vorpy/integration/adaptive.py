"""
Adaptive integrator(s) that control the error on specified quantities by varying the integration timestep.
"""

import abc
import enum
import itertools
import numpy as np
import typing
import vorpy.integration.rungekutta

"""
Design notes
-   The goal is to keep the global error on each quantity within a certain band.
-   This implies that there should be a schedule for how fast each error quantity
    can grow, and it depends on the step size.  So if the error quantity is denoted
    e and its error_band is denoted b = (b.inf, b.sup), then for a timestep t_step, the
    quantity e = e(t) is allowed to grow by about

        np.sqrt(b.inf*b.sup)*t_step / (t_final-t_now)

    And in theory this should allow the global error for e to be about

        np.sqrt(b.inf*b.sup)

    i.e. the geometric mean of the error band.

    Compute a t_step for each error quantity and then take the min of all of them.

-   The integrator is changing t_step up and down basically with every step, which is not ideal.
    Ideally, t_step would change smoothly.  Thus some non-discrete way to decide t_step would be
    better.  Perhaps the (discrete) graph of t_step vs error can be computed for nearby values
    of t_step, and then the "correct" t_step (the one that meets the global error schedule)
    is computed and used.
-   The global error schedule can be relaxed as it goes since the error target is the geometric
    mean of the error band bounds -- don't want it to abort because it can't meet the unnecessarily
    stringent choice of error schedule.  Need to come up with a slightly more flexible criteria for
    this.
-   Because the error computation involves a subtraction, it may be useful to specify how many
    # bits of the error are actually valid.  Perhaps track the number of significant bits in
    each value and computation.
"""

class RealInterval:
    """Defines a closed interval in the real number line."""

    def __init__ (self, inf:float, sup:float) -> None:
        if inf > sup:
            raise ValueError(f'expected inf <= sup, but (inf, sup) was {(inf, sup)}')
        self.inf = inf
        self.sup = sup

    class Membership(enum.Enum):
        BELOW       = enum.auto()
        INTERIOR    = enum.auto()
        ABOVE       = enum.auto()

    def membership (self, x:float) -> 'RealInterval.Membership':
        if not np.isfinite(x):
            raise ValueError(f'x (which is {x}) was expected to be finite')

        if x < self.inf:
            return RealInterval.Membership.BELOW
        elif x > self.sup:
            return RealInterval.Membership.ABOVE
        else:
            return RealInterval.Membership.INTERIOR

    def __repr__ (self) -> str:
        return f'RealInterval({self.inf}, {self.sup})'

class GlobalErrorSchedule:
    def __init__ (self, global_error_band:RealInterval) -> None:
        if global_error_band.inf < 0.0:
            raise ValueError(f'expected global_error_band (which is {global_error_band}) to have a nonnegative lower bound.')

        self.__global_error_band   = global_error_band
        # The geometric mean of the error band bounds is the target error which is what will be used
        # to define the global error schedule.
        self.__global_error_target : float = np.sqrt(global_error_band.inf * global_error_band.sup)

    def global_error_band (self) -> RealInterval:
        return self.__global_error_band

    def global_error_target (self) -> float:
        return self.__global_error_target

    def allowable_error_band_at_time (self, t:float, t_duration:float) -> RealInterval:
        """
        Based on the global error schedule of attempting to attain a global error of the geometric
        mean of error_band inf and sup, this returns the error band that is allowed after a
        particular time.
        """
        proportion = t / t_duration
        return RealInterval(self.__global_error_band.inf*proportion, self.__global_error_target*proportion)

# TODO: Come up with better name for this
class Control:
    def __init__ (
        self,
        *,
        global_error_band:RealInterval,
    ) -> None:
        self.__global_error_schedule = GlobalErrorSchedule(global_error_band)

    @abc.abstractmethod
    def name (self) -> str:
        raise NotImplementedError('subclasses should implement this')

    def global_error_schedule (self) -> GlobalErrorSchedule:
        return self.__global_error_schedule

class ControlledSquaredLTEE(Control):
    NAME = 'Squared LTEE'

    def name (self) -> str:
        return ControlledSquaredLTEE.NAME

class ControlledQuantity(Control):
    def __init__ (
        self,
        *,
        name:str,
        reference_quantity:float, # TODO: Allow arbitrary shapes here, since e.g. angular momentum could be tensor valued
        quantity_evaluator:typing.Callable[[float,np.ndarray],float],
        global_error_band:RealInterval,
    ) -> None:
        super().__init__(global_error_band=global_error_band)

        if len(name) == 0 or name == ControlledSquaredLTEE.NAME:
            raise ValueError(f'name must not be the empty string or "{ControlledSquaredLTEE.NAME}"')

        self.__name                     = name
        self.__reference_quantity       = reference_quantity
        self.__quantity_evaluator       = quantity_evaluator

    def name (self) -> str:
        return self.__name

    def reference_quantity (self) -> float:
        return self.__reference_quantity

    def error (self, t:float, y:np.ndarray) -> float:
        """Computes the error for the given value, using the existing reference_quantity."""
        retval : float = np.abs(self.__quantity_evaluator(t,y) - self.__reference_quantity)
        return retval

class IntegrateVectorFieldResults:
    def __init__ (self, t_v:np.ndarray, y_t:np.ndarray, error_vd:np.ndarray, t_step_v:np.ndarray) -> None:
        # Check the identity claimed for t_v and t_step_v.
        identity_failure_v = (t_v[:-2]+t_step_v[:-1]) - t_v[1:-1]
        print(f'max identity failure: {np.max(np.abs(identity_failure_v))}')
        assert np.max(np.abs(identity_failure_v)) == 0
        print(f'max naive identity failure: {np.max(np.abs(np.diff(t_v[:-1]) - t_step_v[:-1]))}')

        # Sequence of time values, indexed as t_v[i].
        self.t_v        = t_v
        # Sequence (tensor) of parameter values, indexed as y_t[i,J], where i is the time index and
        # is the [multi]index for the parameter type (could be scalar, vector, or tensor).
        self.y_t        = y_t
        # Dictionary of error sequences mapped to their names.  Each error sequence is indexed as error_v[i],
        # where i is the index for t_v.
        self.error_vd   = error_vd
        # Sequence of timestep values, indexed as t_step_v[i], though len(t_step_v) == len(t_v)-1.  Note that
        # this should satisfy t_v[:-1]+t_step_v == t_v[1:] (since each time value is defined as the previous
        # time value plus the current time step), but it will NOT satisfy t_v[1:]-t_v[:-1] == t_step_v due to
        # numerical roundoff error.
        self.t_step_v   = t_step_v

class SalvagedResultsException(Exception):
    pass

def integrate_vector_field (
    *,
    vector_field:typing.Callable[[float,np.ndarray],np.ndarray],
    t_initial:float,
    y_initial:np.ndarray,
    t_final:float,
    controlled_quantity_d:typing.Dict[str,ControlledQuantity],
    controlled_sq_ltee:ControlledSquaredLTEE,
) -> IntegrateVectorFieldResults:
    """
    Integrate a system of ODEs using a variable timestep that's based on bounding the error on
    the specified quantities.  The system of ODEs is defined by the vector field vector_field.  The
    quantities whose error should be controlled are specified by the controlled_quantity_v
    argument.

    vector_field, whose arguments are vector_field(t,y) should return a numpy.ndarray of the same shape and dtype as
    its second input, which should be the same as y_initial.

    The return value is the integrated trajectory, vector of time values, and error values.
    The first index of each of these quantities is the same.  See IntegrateVectorFieldResults.
    """

    # Validate argument values
    if t_final < 0.0:
        raise ValueError(f't_final should be nonnegative, but was actually {t_final}')
    if 'squared LTEE' in controlled_quantity_d.keys():
        raise ValueError('controlled_quantity_d should not contain the reserved key \"squared LTEE\" but it does')

    # These data structures are used to accumulate the return values
    t_v         : typing.List[float]                    = []
    y_tv        : typing.List[np.ndarray]               = []
    error_vd    : typing.Dict[str,typing.List[float]]   = {name:[] for name in controlled_quantity_d}
    error_vd[ControlledSquaredLTEE.NAME]                = []
    t_step_v    : typing.List[float]                    = []

    def add_sample (t:float, y:np.ndarray, error_d:typing.Dict[str,float], dt_o:typing.Optional[float]) -> None:
        if error_d.keys() != error_vd.keys():
            raise TypeError(f'expected error_d to have keys {error_vd.keys()} but it had keys {error_d.keys()}')
        t_v.append(t)
        y_tv.append(np.copy(y))
        for name,error in error_d.items():
            error_vd[name].append(error)
        if dt_o is not None:
            t_step_v.append(dt_o)

    def make_error_d (controlled_quantity_error_d:typing.Dict[str,float], sq_ltee:float) -> typing.Dict[str,float]:
        if controlled_quantity_error_d.keys() != controlled_quantity_d.keys():
            raise TypeError(f'expected controlled_quantity_error_d to have keys {controlled_quantity_d.keys()} but it had keys {controlled_quantity_error_d.keys()}')
        error_d = dict(**controlled_quantity_error_d)
        error_d[ControlledSquaredLTEE.NAME] = sq_ltee
        return error_d

    # No t_step for first sample
    add_sample(t_initial, y_initial, make_error_d({name:0.0 for name in controlled_quantity_d}, 0.0), None)

    # Must use an integrator that has local truncation error estimation.
    integrator = vorpy.integration.rungekutta.RungeKuttaFehlberg_4_5(vector_field=vector_field, parameter_shape=y_initial.shape)
    integrator.set_inputs(t_initial, y_initial)

    # Make a dict of all the controls to make certain things easier
    control_d : typing.Dict[str,Control] = dict(**controlled_quantity_d, **{controlled_sq_ltee.name():controlled_sq_ltee})

    if False:
        def log_message (message:str) -> None:
            print(message)
    else:
        def log_message (message:str) -> None:
            pass

    try:
        t_step = 1.0e-2 # Arbitrary for now
        t_duration = t_final - t_initial
        iteration_index = 0
        while integrator.t_now < t_final:
            log_message('\n-------------------------------------------------')
            log_message(f'iteration_index = {iteration_index}')

            if False:
                def log_t_step_message (message:str) -> None:
                    log_message(f'    {message}')
            else:
                def log_t_step_message (message:str) -> None:
                    pass

            # TODO: Don't use a hard and fast rule for the local controlled quantities, because it can
            # prevent them from running past a certain point.  maybe instead try to minimize a quadratic
            # function in the errors, whose minimum occurs at the "ideal error values"
            # IDEA: Use a separate bound for global error compared to local error for controlled quantities.
            # This way, the global error can be used to make large-scale decisions, such as when the
            # integration needs to use an overall smaller or bigger timestep to meet the global error bound.
            # IDEA: Try to meet a schedule for how the global errors for different CQs evolve.  For example,
            # require that the global error grows linearly such that it still comes in under the error bound.
            # This would require keeping each local CQ error within some computed bound for each step.

            # Run the integrator, updating the timestep until the error bounds are satisfied.
            break_after_computing_step = False
            t_step_is_bad = False
            t_step_iteration_index = 0
            can_go_in_direction_o = 0
            while True:
                log_t_step_message(f'---- t_step-finding iteration {t_step_iteration_index}')
                if integrator.t_now + t_step == integrator.t_now:
                    log_t_step_message('t_step is too small (THIS PROBABLY MEANS THE GLOBAL ERROR SCHEDULE IS NOT AGGRESSIVE ENOUGH); breaking')
                    t_step_is_bad = True
                    break
                if t_step_iteration_index > 1000:
                    log_t_step_message(f't_step_iteration_index exceeded limit; breaking')
                    t_step_is_bad = True
                    break
                log_t_step_message(f't_step_iteration_index  {t_step_iteration_index}; t_step = {t_step}')
                # Make sure t_step wouldn't exceed t_final
                if t_step > t_final - integrator.t_now:
                    t_step = t_final - integrator.t_now
                    assert integrator.t_now + t_step >= t_final, 't_step needs to be adjusted more carefully due to roundoff error in the subtraction'
                # Run the integrator one step
                integrator.step(t_step)
                # Analyze the various error band excesses
                error_d = make_error_d(
                    {
                        name:cq.error(integrator.t_next, integrator.y_next)
                        for name,cq in controlled_quantity_d.items()
                    },
                    integrator.ltee_squared
                )
                allowable_error_band_d = {
                    name:c.global_error_schedule().allowable_error_band_at_time(integrator.t_next, t_duration)
                    for name,c in control_d.items()
                }
                error_band_membership_d = {
                    name:allowable_error_band.membership(error_d[name])
                    for name,allowable_error_band in allowable_error_band_d.items()
                }
                log_t_step_message(f'error_d = {error_d}')
                log_t_step_message(f'allowable_error_band_d = {allowable_error_band_d}')
                log_t_step_message(f'error_band_membership_d = {error_band_membership_d}')

                if break_after_computing_step:
                    break

                if can_go_in_direction_o == 0:
                    # If any bands are exceeded ABOVE, then set can_go_in_direction_o to a negative value
                    if RealInterval.Membership.ABOVE in error_band_membership_d.values():
                        can_go_in_direction_o = -1
                    # Otherwise if any bands are exceeded BELOW, then set can_go_in_direction_o to a positive value
                    elif RealInterval.Membership.BELOW in error_band_membership_d.values():
                        can_go_in_direction_o = 1
                    # Otherwise there's no problem, so no update is necessary, so break out of this
                    # t_step-determining loop.
                    else:
                        break

                #dt_adjustment_power = 1/2
                dt_adjustment_power = 1/4

                log_t_step_message(f'can_go_in_direction_o was set to {can_go_in_direction_o}')
                assert can_go_in_direction_o != 0
                if can_go_in_direction_o < 0:
                    if RealInterval.Membership.ABOVE in error_band_membership_d.values():
                        t_step *= 0.5**dt_adjustment_power
                        log_t_step_message(f'decreasing t_step to {t_step}')
                    else:
                        break
                else:
                    assert can_go_in_direction_o > 0
                    # ABOVE is unacceptable, so if that occured, then back off one and use it.
                    if RealInterval.Membership.ABOVE in error_band_membership_d.values():
                        t_step *= 0.5**dt_adjustment_power
                        break_after_computing_step = True
                        log_t_step_message(f'decreasing t_step to {t_step} and using that one')
                    elif RealInterval.Membership.BELOW in error_band_membership_d.values():
                        t_step *= 2.0**dt_adjustment_power
                        log_t_step_message(f'increasing t_step to {t_step}')
                    else:
                        break

                t_step_iteration_index += 1

            if t_step_is_bad:
                break

            log_message(f't = {integrator.t_now}; using t_step {t_step}, error_d = {error_d}')

            # Store the sample for the return value(s)
            add_sample(integrator.t_next, integrator.y_next, error_d, t_step)
            # Set t_now, y_now to be t_next, y_next so we can compute the next step.
            log_message(f'integrator.get_outputs = {integrator.get_outputs()}')
            integrator.set_inputs(*integrator.get_outputs())
            iteration_index += 1

    except KeyboardInterrupt as e:
        print('Caught KeyboardInterrupt -- returning existing results.')
        #raise SalvagedResultsException(construct_results()) from e
    finally:
        print('stuff')

    print(f'returning results')

    return IntegrateVectorFieldResults(
        t_v=np.array(t_v),
        y_t=np.array(y_tv),
        error_vd={name:np.array(error_v) for name,error_v in error_vd.items()},
        t_step_v=np.array(t_step_v),
    )

if __name__ == '__main__':
    import sympy as sp
    import vorpy
    import vorpy.symbolic
    import vorpy.symplectic

    np.set_printoptions(precision=20)

    # Define the Kepler problem and use it to test the integrator
    def do_kepler_problem () -> None:

        def phase_space_coordinates () -> np.ndarray:
            return np.array(sp.var('x,y,p_x,p_y')).reshape(2,2)

        def K (p:np.ndarray) -> typing.Any: # Not sure how to annotate a general scalar
            return np.dot(p.flat, p.flat) / 2

        def U (q:np.ndarray) -> typing.Any:
            return -1 / sp.sqrt(np.dot(q.flat, q.flat))

        def H (qp:np.ndarray) -> typing.Any:
            """Total energy -- should be conserved."""
            return K(qp[1,...]) + U(qp[0,...])

        def p_theta (qp:np.ndarray) -> typing.Any:
            """Angular momentum -- should be conserved."""
            x,y,p_x,p_y = qp.reshape(-1)
            return x*p_y - y*p_x

        # Determine the Hamiltonian vector field of H.
        qp = phase_space_coordinates()
        X_H = vorpy.symplectic.symplectic_gradient_of(H(qp), qp)
        print(f'X_H:\n{X_H}')
        print('X_H lambdification')
        X_H_fast = vorpy.symbolic.lambdified(X_H, qp, replacement_d={'array':'np.array', 'dtype=object':'dtype=np.float64'}, verbose=True)

        print('H lambdification')
        H_fast = vorpy.symbolic.lambdified(H(qp), qp, replacement_d={'sqrt':'np.sqrt'}, verbose=True)
        print('p_theta lambdification')
        p_theta_fast = vorpy.symbolic.lambdified(p_theta(qp), qp, verbose=True)

        t_initial = 0.0
        qp_initial = np.array([[1.0,0.0],[0.0,0.01]])
        H_initial = H_fast(qp_initial)
        p_theta_initial = p_theta_fast(qp_initial)

        print(f'H_initial = {H_initial}')
        print(f'p_theta_initial = {p_theta_initial}')

        H_cq = ControlledQuantity(
            name='H',
            reference_quantity=H_initial,
            global_error_band=RealInterval(1.0e-10, 1.0e-6),
            quantity_evaluator=(lambda t,qp:typing.cast(float, H_fast(qp))),
        )
        p_theta_cq = ControlledQuantity(
            name='p_theta',
            reference_quantity=p_theta_initial,
            global_error_band=RealInterval(1.0e-10, 1.0e-6),
            quantity_evaluator=(lambda t,qp:typing.cast(float, p_theta_fast(qp))),
        )
        controlled_sq_ltee = ControlledSquaredLTEE(
            global_error_band=RealInterval(1.0e-12**2, 1.0e-6**2),
        )

        try:
            results = integrate_vector_field(
                vector_field=(lambda t,qp:X_H_fast(qp)),
                t_initial=t_initial,
                y_initial=qp_initial,
                t_final=10.0,
                controlled_quantity_d={
                    'H abs error':H_cq,
                    'p_theta abs error':p_theta_cq,
                },
                controlled_sq_ltee=controlled_sq_ltee,
            )
            #print(f'results = {results}')
        except SalvagedResultsException as e:
            print('got SalvagedResultsException')
            results = e.args[0]
        finally:
            H_v = vorpy.apply_along_axes(H_fast, (1,2), (results.y_t,))
            #H_error_v = vorpy.apply_along_axes(lambda qp:H_cq.error(0.0,qp), (1,2), (results.y_t,))
            #print(f'H_v = {H_v}')
            #print(f'H_error_v = {H_error_v}')

            p_theta_v = vorpy.apply_along_axes(p_theta_fast, (1,2), (results.y_t,))
            #p_theta_error_v = vorpy.apply_along_axes(lambda qp:p_theta_cq.error(0.0,qp), (1,2), (results.y_t,))
            #print(f'p_theta_v = {p_theta_v}')
            #print(f'p_theta_error_v = {p_theta_error_v}')

            import matplotlib.pyplot as plt

            def plot_stuff () -> None:
                row_count   = 2
                col_count   = 4
                size        = 5
                fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

                axis = axis_vv[0][0]
                axis.set_title('position')
                axis.set_aspect('equal')
                axis.plot(results.y_t[:,0,0], results.y_t[:,0,1], '.')

                #axis = axis_vv[1][0]
                #axis.set_title('momentum')
                #axis.set_aspect('equal')
                #axis.plot(results.y_t[:,1,0], results.y_t[:,1,1], '.')

                axis = axis_vv[1][0]
                axis.set_title('blue:(t,x), orange:(t,y)')
                axis.plot(results.t_v, results.y_t[:,0,:], '.')
                #axis.plot(results.t_v, results.y_t[:,0,1], '.')

                axis = axis_vv[0][1]
                axis.set_title('H abs error')
                axis.semilogy(results.t_v, results.error_vd['H abs error'], '.')
                axis.axhline(H_cq.global_error_schedule().global_error_band().inf, color='green')
                axis.axhline(H_cq.global_error_schedule().global_error_target(), color='blue')
                axis.axhline(H_cq.global_error_schedule().global_error_band().sup, color='red')

                axis = axis_vv[1][1]
                axis.set_title('p_theta abs error')
                axis.semilogy(results.t_v, results.error_vd['p_theta abs error'], '.')
                axis.axhline(p_theta_cq.global_error_schedule().global_error_band().inf, color='green')
                axis.axhline(p_theta_cq.global_error_schedule().global_error_target(), color='blue')
                axis.axhline(p_theta_cq.global_error_schedule().global_error_band().sup, color='red')

                axis = axis_vv[0][2]
                axis.set_title(ControlledSquaredLTEE.NAME)
                axis.semilogy(results.t_v, results.error_vd[ControlledSquaredLTEE.NAME], '.')
                axis.axhline(controlled_sq_ltee.global_error_schedule().global_error_band().inf, color='green')
                axis.axhline(controlled_sq_ltee.global_error_schedule().global_error_target(), color='blue')
                axis.axhline(controlled_sq_ltee.global_error_schedule().global_error_band().sup, color='red')

                axis = axis_vv[0][3]
                axis.set_title('timestep')
                axis.semilogy(results.t_v[1:], results.t_step_v, '.')

                axis = axis_vv[1][3]
                axis.set_title('delta timestep')
                delta_timestep = np.diff(results.t_step_v)
                axis.plot(results.t_v[1:-1], np.sign(delta_timestep)*np.log10(np.abs(delta_timestep)), '.')

                fig.tight_layout()
                filename = 'integrator.png'
                plt.savefig(filename, bbox_inches='tight')
                print('wrote to file "{0}"'.format(filename))
                # VERY important to do this -- otherwise your memory will slowly fill up!
                # Not sure which one is actually sufficient -- apparently none of them are, YAY!
                plt.clf()
                plt.close(fig)
                plt.close('all')
                del fig
                del axis_vv

            plot_stuff()

    def do_three_body_problem () -> None:
        def phase_space_coordinates () -> np.ndarray:
            return np.array(sp.var('x_0,y_0,x_1,y_1,x_2,y_2,u_0,v_0,u_1,v_1,u_2,v_2')).reshape(2,3,2)

        def K (p:np.ndarray) -> typing.Any: # Not sure how to annotate a general scalar
            return np.dot(p.flat, p.flat) / 2

        def U (q:np.ndarray) -> typing.Any:
            total = 0
            for i in range(3):
                for j in range(i):
                    delta = q[i,:] - q[j,:]
                    total += -1 / sp.sqrt(np.dot(delta, delta))
            return total

        def H (qp:np.ndarray) -> typing.Any:
            """Total energy -- should be conserved."""
            return K(qp[1,...]) + U(qp[0,...])

        def p_theta (qp:np.ndarray) -> typing.Any:
            """Angular momentum -- should be conserved."""
            total = 0
            for i in range(3):
                x,y,u,v = qp[:,i,:].reshape(-1)
                total += x*v - y*u
            return total

        def norm_deltas (qp:np.ndarray) -> np.ndarray:
            """Returns the expression defining the distance between bodies i and j for ij = 01, 02, 12 in that order."""
            def norm_delta_ij (i:int, j:int) -> typing.Any:
                q = qp[0,...]
                q_i = q[i,:]
                q_j = q[j,:]
                delta = q_i - q_j
                return sp.sqrt(np.dot(delta, delta))
            return np.array([norm_delta_ij(0,1), norm_delta_ij(0,2), norm_delta_ij(1,2)])

        #def norm_delta (i:int, j:int) -> typing.Callable[[np.ndarray],typing.Any]:
            #"""Returns the expression defining the distance between bodies i and j."""
            #def norm_delta_ij (qp:np.ndarray) -> typing.Any:
                #q = qp[0,...]
                #q_i = q[i,:]
                #q_j = q[j,:]
                #delta = q_i - q_j
                #return sp.sqrt(np.dot(delta, delta))
            #return norm_delta_ij

        def body_vertex_angles (qp:np.ndarray) -> np.ndarray:
            """Returns the angle of the vertex of the shape triangle that body i sits at for i = 0, 1, 2."""
            def body_vertex_angle_i (i:int) -> typing.Any:
                q = qp[0,...]
                # Determine the index of each of the other two bodies.
                j = (i+1)%3
                k = (i+2)%3
                q_i = q[i,:]
                q_j = q[j,:]
                q_k = q[k,:]
                delta_ij = q_j - q_i
                delta_ik = q_k - q_i
                norm_squared_delta_ij = np.dot(delta_ij, delta_ij)
                norm_squared_delta_ik = np.dot(delta_ik, delta_ik)
                cos_angle = np.dot(delta_ij, delta_ik) / sp.sqrt(norm_squared_delta_ij*norm_squared_delta_ik)
                return sp.acos(cos_angle)
            return np.array([body_vertex_angle_i(0), body_vertex_angle_i(1), body_vertex_angle_i(2)])

        def shape_area (qp:np.ndarray) -> typing.Any:
            """Area of the parallelopiped spanned by the segments 01 and 02."""
            q = qp[0,...]
            delta_01 = q[1,:] - q[0,:]
            delta_02 = q[2,:] - q[0,:]
            return np.cross(delta_01, delta_02)[()]

        # Determine the Hamiltonian vector field of H.
        qp = phase_space_coordinates()
        X_H = vorpy.symplectic.symplectic_gradient_of(H(qp), qp.reshape(2,6)).reshape(2,3,2)
        print(f'X_H:\n{X_H}')

        replacement_d = {
            'array'         :'np.array',
            'dtype=object'  :'dtype=np.float64',
            'sqrt'          :'np.sqrt',
            'acos'          :'np.arccos',
        }

        print('X_H lambdification')
        X_H_fast = vorpy.symbolic.lambdified(X_H, qp, replacement_d=replacement_d, verbose=True)

        print('H lambdification')
        H_fast = vorpy.symbolic.lambdified(H(qp), qp, replacement_d=replacement_d, verbose=True)
        print('p_theta lambdification')
        p_theta_fast = vorpy.symbolic.lambdified(p_theta(qp), qp, replacement_d=replacement_d, verbose=True)
        print('shape_area lambdification')
        shape_area_fast = vorpy.symbolic.lambdified(shape_area(qp), qp, replacement_d=replacement_d, verbose=True)
        print('norm_deltas lambdification')
        norm_deltas_fast = vorpy.symbolic.lambdified(norm_deltas(qp), qp, replacement_d=replacement_d, verbose=True)
        print('body_vertex_angles lambdification')
        body_vertex_angles_fast = vorpy.symbolic.lambdified(body_vertex_angles(qp), qp, replacement_d=replacement_d, verbose=True)

        t_initial = 0.0
        qp_initial = np.array([
            [[-1.0,0.0],[0.0,1.0],[1.0,0.0]],
            [[ 0.0,-0.3],[-0.3,0.0],[0.1,0.2]], # zero momentum to begin with
        ])
        H_initial = H_fast(qp_initial)
        p_theta_initial = p_theta_fast(qp_initial)

        print(f'H_initial = {H_initial}')
        print(f'p_theta_initial = {p_theta_initial}')

        H_cq = ControlledQuantity(
            name='H',
            reference_quantity=H_initial,
            #global_error_band=RealInterval(1.0e-10, 1.0e-6),
            global_error_band=RealInterval(1.0e-8, 1.0e-5),
            quantity_evaluator=(lambda t,qp:typing.cast(float, H_fast(qp))),
        )
        p_theta_cq = ControlledQuantity(
            name='p_theta',
            reference_quantity=p_theta_initial,
            #global_error_band=RealInterval(1.0e-10, 1.0e-6),
            global_error_band=RealInterval(1.0e-8, 1.0e-5),
            quantity_evaluator=(lambda t,qp:typing.cast(float, p_theta_fast(qp))),
        )
        controlled_sq_ltee = ControlledSquaredLTEE(
            global_error_band=RealInterval(1.0e-12**2, 1.0e-6**2),
        )

        try:
            results = integrate_vector_field(
                vector_field=(lambda t,qp:X_H_fast(qp)),
                t_initial=t_initial,
                y_initial=qp_initial,
                t_final=10.0,
                controlled_quantity_d={
                    'H abs error':H_cq,
                    'p_theta abs error':p_theta_cq,
                },
                controlled_sq_ltee=controlled_sq_ltee,
            )
            #print(f'results = {results}')
        except SalvagedResultsException as e:
            print('got SalvagedResultsException')
            results = e.args[0]
        finally:
            H_v = vorpy.apply_along_axes(H_fast, (1,2,3), (results.y_t,))
            #H_error_v = vorpy.apply_along_axes(lambda qp:H_cq.error(0.0,qp), (1,2), (results.y_t,))
            #print(f'H_v = {H_v}')
            #print(f'H_error_v = {H_error_v}')

            p_theta_v = vorpy.apply_along_axes(p_theta_fast, (1,2,3), (results.y_t,))
            #p_theta_error_v = vorpy.apply_along_axes(lambda qp:p_theta_cq.error(0.0,qp), (1,2), (results.y_t,))
            #print(f'p_theta_v = {p_theta_v}')
            #print(f'p_theta_error_v = {p_theta_error_v}')

            shape_area_v = vorpy.apply_along_axes(shape_area_fast, (1,2,3), (results.y_t,))

            norm_deltas_t = vorpy.apply_along_axes(norm_deltas_fast, (1,2,3), (results.y_t,))
            print(f'norm_deltas_t = {norm_deltas_t}')

            body_vertex_angles_t = vorpy.apply_along_axes(body_vertex_angles_fast, (1,2,3), (results.y_t,))
            print(f'body_vertex_angles_t = {body_vertex_angles_t}')

            sign_change_v = shape_area_v[:-1]*shape_area_v[1:]
            zero_crossing_index_v = (sign_change_v < 0)
            zero_crossing_v = results.t_v[:-1][zero_crossing_index_v]

            print(f'body_vertex_angles_t[:-1][zero_crossing_index_v,:] = {body_vertex_angles_t[:-1][zero_crossing_index_v,:]}')

            syzygy_v = np.argmax(body_vertex_angles_t[:-1][zero_crossing_index_v,:], axis=1)
            print(f'syzygy_v = {syzygy_v}')

            import matplotlib.pyplot as plt

            def plot_stuff () -> None:
                row_count   = 2
                col_count   = 4
                size        = 10
                fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

                axis = axis_vv[0][0]
                axis.set_title('position (bodies 0,1,2 are R,G,B resp.)')
                axis.set_aspect('equal')
                axis.plot(results.y_t[:,0,0,0], results.y_t[:,0,0,1], color='red')
                axis.plot(results.y_t[:,0,1,0], results.y_t[:,0,1,1], color='green')
                axis.plot(results.y_t[:,0,2,0], results.y_t[:,0,2,1], color='blue')

                axis = axis_vv[1][0]
                axis.set_title('norm delta ij (color is combination of body color)')
                axis.semilogy(results.t_v, norm_deltas_t[:,0], color='yellow')
                axis.semilogy(results.t_v, norm_deltas_t[:,1], color='magenta')
                axis.semilogy(results.t_v, norm_deltas_t[:,2], color='cyan')
                for zero_crossing in zero_crossing_v:
                    axis.axvline(zero_crossing, color='black', alpha=0.5)

                #axis = axis_vv[1][0]
                #axis.set_title('momentum')
                #axis.set_aspect('equal')
                #axis.plot(results.y_t[:,1,0], results.y_t[:,1,1], '.')

                #axis = axis_vv[1][0]
                #axis.set_title('blue:(t,x), orange:(t,y)')
                #axis.plot(results.t_v, results.y_t[:,0,:], '.')
                ##axis.plot(results.t_v, results.y_t[:,0,1], '.')

                axis = axis_vv[0][1]
                axis.set_title('H abs error')
                axis.semilogy(results.t_v, results.error_vd['H abs error'], '.')
                axis.axhline(H_cq.global_error_schedule().global_error_band().inf, color='green')
                axis.axhline(H_cq.global_error_schedule().global_error_target(), color='blue')
                axis.axhline(H_cq.global_error_schedule().global_error_band().sup, color='red')

                axis = axis_vv[1][1]
                axis.set_title('p_theta abs error')
                axis.semilogy(results.t_v, results.error_vd['p_theta abs error'], '.')
                axis.axhline(p_theta_cq.global_error_schedule().global_error_band().inf, color='green')
                axis.axhline(p_theta_cq.global_error_schedule().global_error_target(), color='blue')
                axis.axhline(p_theta_cq.global_error_schedule().global_error_band().sup, color='red')

                axis = axis_vv[0][2]
                axis.set_title(ControlledSquaredLTEE.NAME)
                axis.semilogy(results.t_v, results.error_vd[ControlledSquaredLTEE.NAME], '.')
                axis.axhline(controlled_sq_ltee.global_error_schedule().global_error_band().inf, color='green')
                axis.axhline(controlled_sq_ltee.global_error_schedule().global_error_target(), color='blue')
                axis.axhline(controlled_sq_ltee.global_error_schedule().global_error_band().sup, color='red')

                axis = axis_vv[1][2]
                axis.set_title('shape_area (vertical line indicates syzygy)')
                axis.plot(results.t_v, shape_area_v)#, '.')
                axis.axhline(0)
                for zero_crossing in zero_crossing_v:
                    axis.axvline(zero_crossing, color='black', alpha=0.5)

                axis = axis_vv[0][3]
                axis.set_title('timestep')
                axis.semilogy(results.t_v[1:], results.t_step_v, '.')

                axis = axis_vv[1][3]
                axis.set_title('body vertex angles (bodies 0,1,2 are R,G,B resp.)')
                axis.plot(results.t_v, body_vertex_angles_t[:,0], color='red')
                axis.plot(results.t_v, body_vertex_angles_t[:,1], color='green')
                axis.plot(results.t_v, body_vertex_angles_t[:,2], color='blue')
                for zero_crossing in zero_crossing_v:
                    axis.axvline(zero_crossing, color='black', alpha=0.2)

                #axis = axis_vv[1][3]
                #axis.set_title('delta timestep (signed log_10 plot)')
                #delta_timestep = np.diff(results.t_step_v)
                #axis.plot(results.t_v[1:-1], np.sign(delta_timestep)*np.log10(np.abs(delta_timestep)), '.')

                fig.tight_layout()
                filename = 'threebody.png'
                plt.savefig(filename, bbox_inches='tight')
                print('wrote to file "{0}"'.format(filename))
                # VERY important to do this -- otherwise your memory will slowly fill up!
                # Not sure which one is actually sufficient -- apparently none of them are, YAY!
                plt.clf()
                plt.close(fig)
                plt.close('all')
                del fig
                del axis_vv

            plot_stuff()

    #do_kepler_problem()
    do_three_body_problem()

