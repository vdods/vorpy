"""
Implements explicit Runge-Kutta integration methods, of ordinary (non-error-estimating) and error-estimating types.
"""

import abc
import numpy as np
import typing
import vorpy.tensor

class RungeKutta(metaclass=abc.ABCMeta):
    """
    References:
    -   Wikipedia RK article - https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    -   List of RK methods - https://ipfs.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/List_of_Runge%E2%80%93Kutta_methods.html
    -   A Tenth-Order Runge-Kutta Method with Error Estimate by T. Feagin - http://sce.uhcl.edu/feagin/courses/rk10.pdf
    -   An Explicit Sixth-Order Runge-Kutta Formula By H. A. Luther - https://www.ams.org/journals/mcom/1968-22-102/S0025-5718-68-99876-1/S0025-5718-68-99876-1.pdf
    -   Appendix A; Runge-Kutta Methods - https://www.uni-muenster.de/imperia/md/content/physik_tp/lectures/ss2017/numerische_Methoden_fuer_komplexe_Systeme_II/rkm-1.pdf
    """

    @classmethod
    @abc.abstractmethod
    def order (cls) -> int:
        """
        Should return the order of this method.  If a method has order p, then its local truncation error
        will be on the order of O(dt^(p+1)).  Note that there is no simple relationship between order
        and stage count.

        From https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods

            In general, if an explicit s-stage Runge–Kutta method has order p, then it can be proven that
            the number of stages must satisfy s >= p, and if p >= 5, then s >= p+1.  However, it is not
            known whether these bounds are sharp in all cases.
        """
        raise NotImplementedError('subclass must implement this in order to use it')

    # Note: @abc.abstractmethod should be the innermost decorator;
    # see https://docs.python.org/3/library/abc.html#abc.abstractmethod
    @classmethod
    @abc.abstractmethod
    def a (cls) -> np.ndarray:
        """
        Returns the `a` part of the Butcher tableau of this RK method.
        See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Adaptive_Runge%E2%80%93Kutta_methods

        Return value should have shape
        """
        raise NotImplementedError('subclass must implement this in order to use it')

    @classmethod
    @abc.abstractmethod
    def b (cls) -> np.ndarray:
        """
        Returns the `b` part of the Butcher tableau of this RK method.
        See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Adaptive_Runge%E2%80%93Kutta_methods
        """
        raise NotImplementedError('subclass must implement this in order to use it')

    @classmethod
    def b_star (cls) -> np.ndarray:
        """
        Returns the `b*` part of the Butcher tableau of this RK method.
        See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Adaptive_Runge%E2%80%93Kutta_methods

        Note that a non-embedded Runge-Kutta method does not need to implement this.
        """
        raise NotImplementedError('subclass must implement this in order to use it')

    @classmethod
    @abc.abstractmethod
    def c (cls) -> np.ndarray:
        """
        Returns the `c` part of the Butcher tableau of this RK method.
        See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Adaptive_Runge%E2%80%93Kutta_methods
        """
        raise NotImplementedError('subclass must implement this in order to use it')

    @classmethod
    @abc.abstractmethod
    def is_explicit_method (cls) -> bool:
        """
        Should return true if this is an explicit method (meaning there are certain constraints on the
        Butcher tableau).  Default is False (i.e. no constraint).

        See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods
        """
        return False

    @classmethod
    @abc.abstractmethod
    def is_embedded_method (cls) -> bool:
        """
        Should return true if this is an embedded method (meaning it uses a secondary, higher-order method
        to estimate the local truncation error).  Default is False (i.e. no secondary, higher-order method).

        See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Adaptive_Runge%E2%80%93Kutta_methods
        """
        return False

    @classmethod
    def validate_method_definition (cls) -> None:
        """
        Will raise an exception if there is any inconsistency in the definition of a, b, c (i.e. the Butcher
        tableau) of this method.  If cls.is_explicit_method returns True, then it will require that a is strictly
        lower-triangular.

        If all checks pass, no exception will be raised.
        """
        a = cls.a()
        if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
            raise TypeError(f'expected a to be a square matrix (but a.shape was {a.shape}')

        stage_count = a.shape[0]
        order = cls.order()
        if order >= 5:
            if not (stage_count >= order+1):
                raise ValueError(f'For a Runge-Kutta method of order >= 5, the number of stages must be >= order+1 (but in this case, order = {order} and stage_count = {stage_count}')
        else:
            if not (stage_count >= order):
                raise ValueError(f'For a Runge-Kutta method of order < 5, the number of stages must be >= order (but in this case, order = {order} and stage_count = {stage_count}')

        if cls.is_explicit_method():
            for row in range(stage_count):
                for col in range(row,stage_count):
                    if a[row,col] != 0.0:
                        raise ValueError(f'expected a to be strictly lower-triangular because cls.is_explicit_method() was True (but a was\n{a}')

        b = cls.b()
        if len(b.shape) != 1 or b.shape[0] != stage_count:
            raise TypeError(f'expected b to be a vector having dimension {stage_count} (but b.shape was {b.shape})')

        if cls.is_embedded_method():
            b_star = cls.b_star()
            if len(b_star.shape) != 1 or b_star.shape[0] != stage_count:
                raise TypeError(f'expected b_star to be a vector having dimension {stage_count} (but b_star.shape was {b_star.shape})')

            # The following 2 checks aren't rigorously backed up, but are just from an observation that
            # fewer stages seem to imply lower order.  Thus b (which should produce the integrator of
            # order equal to cls.order()) should have a zero at least in its last component.

            if b_star[-1] == 0.0:
                raise ValueError(f'expected b_star to have a non-zero final component (but b_star was {b_star}')

            if b[-1] != 0.0:
                raise ValueError(f'expected b to have a zero final component (but b was {b}')

        c = cls.c()
        if len(c.shape) != 1 or c.shape[0] != stage_count:
            raise TypeError(f'expected c to be a vector having dimension {stage_count} (but c.shape was {c.shape})')

        if cls.is_explicit_method():
            if c[0] != 0.0:
                raise ValueError(f'expected c[0] to be zero because cls.is_explicit_method() was true (but c[0] was {c[0]}')

    @classmethod
    def order (cls) -> int:
        """
        Returns the order of the method, meaning that the local truncation error is on the order of O(dt^(order+1)),
        and the total accumulated error is on the order of O(dt^order).
        """
        cls.validate_method_definition()
        return __order_of_vector(cls.b())

    @classmethod
    def _stage_count (cls) -> int:
        cls.validate_method_definition()
        return cls.a().shape[0]

class RungeKutta_Explicit(RungeKutta):
    """
    NOTE: For the time being, it is assumed that the computed integration step of an embedded integrator
    will be the lower-order value, since the higher-order value is ostensibly used to estimate the local
    truncation error.

    TODO: Write tests that verify that error is of the claimed order.
    TODO: b - b_star is constant (per method), so pre-compute this.
    TODO: Depending on what the semantics of b_star actually are (which one of b or b_star is used to
          produce the result), maybe rename this to b_embedded?
    TODO: implement estimation of global error (presumably it's the sum of local truncation error)
    """

    def __init__ (
        self,
        *,
        vector_field:typing.Callable[[float,np.ndarray],np.ndarray],
        parameter_shape:typing.Sequence[int],
    ) -> None:
        if not all(s >= 0 for s in parameter_shape):
            raise ValueError(f'parameter_shape must have all nonnegative components (but was actually {parameter_shape}')

        self.validate_method_definition()

        self.__vector_field         = vector_field
        self.__parameter_shape      = parameter_shape
        self.__parameter_dimension  = vorpy.tensor.dimension_of_shape(parameter_shape)
        self.__stage_count          = self._stage_count()
        # Create and keep some np.ndarray instances for intermediate and result computations in order to avoid
        # memory allocation during integration.
        self.__k                    = np.zeros((self.__stage_count, self.__parameter_dimension), dtype=np.float64)
        # This is the time value input to the integrator's step function.
        self.t_now                  = 0.0
        # This is the parameter value input to the integrator's step function.
        self.y_now                  = np.zeros(parameter_shape, dtype=np.float64)
        # This is the time value output to the integrator's step function (the result is stored here).
        self.t_next                 = 0.0
        # This is the parameter value  output to the integrator's step function (the result is stored here).
        self.y_next                 = np.zeros(parameter_shape, dtype=np.float64)
        # If this is an embedded method, create an array for storage of the [square of the] local truncation
        # error estimate.  We use the square in order to avoid taking a square root during integration.
        if self.is_embedded_method():
            self.ltee_squared       = np.nan

    @classmethod
    def is_explicit_method (cls) -> bool:
        return True

    def set_inputs (self, t:float, y:np.ndarray) -> None:
        self.t_now      = t
        self.y_now[:]   = y

    def get_outputs (self) -> typing.Tuple[float, np.ndarray]:
        return self.t_next, self.y_next

    def get_local_truncation_error_estimate (self) -> float:
        """
        Returns the local truncation error estimate of the last call to step.  Note that this function
        calls numpy.sqrt, since the square of the LTEE is what is computed and stored, in order to
        avoid a call to sqrt during the integration step.  To access the squared LTEE, just use the
        ltee_squared attribute directly.
        """
        return np.sqrt(self.ltee_squared)

    def step (self, dt:float) -> None:
        """
        Integrates the initial conditions (t,y) using timestep dt and RK method defined by a, b, c (i.e.
        the Butcher tableau of the method).

        Stores the updated t and y values in self.t and self.y.

        Returns self.t, self.y for convenience.

        Reference:
        -   https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods
        """

        a = self.a()
        b = self.b()
        if self.is_embedded_method():
            b_star = self.b_star()
        c = self.c()

        # Because this is an explicit method, a[0,:] and c[0] are identically zero, so the first iteration
        # reduces to a simpler form.  Flatten the result in order to make the index computations involving
        # __k simple.
        self.__k[0,:]       = self.__vector_field(self.t_now, self.y_now).reshape(-1)

        # Do the rest of the iterations using the general form.
        for i in range(1, self.__stage_count):
            # Flatten the result in order to be able to assign to __k (which is a flattened parameter_shape).
            self.__k[i,:]   = self.__vector_field(
                self.t_now + dt*c[i],
                self.y_now + dt*np.einsum('i,ij->j', a[i,0:i], self.__k[0:i,:]).reshape(*self.__parameter_shape),
            ).reshape(-1)

        self.t_next         = self.t_now + dt
        self.y_next[:]      = self.y_now + dt*np.einsum('i,ij->j', b, self.__k).reshape(*self.__parameter_shape)
        if self.is_embedded_method():
            self.ltee_squared = (dt**2) * np.sum(np.einsum('i,ij->j', b - b_star, self.__k)**2)

class RungeKutta_4(RungeKutta_Explicit):
    """
    The original Runge-Kutta 4 method -- a 4th order method.  Does not do any local truncation error estimation.

    Reference:
    -   https://ipfs.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/List_of_Runge%E2%80%93Kutta_methods.html
    """

    # Define the Butcher tableau using class variables, so new np.ndarrays aren't created during the step function.
    __a = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    __b = np.array([1/6, 1/3, 1/3, 1/6])
    __c = np.array([0.0, 0.5, 0.5, 1.0])

    @classmethod
    def order (cls) -> int:
        return 4

    @classmethod
    def a (cls) -> np.ndarray:
        return cls.__a

    @classmethod
    def b (cls) -> np.ndarray:
        return cls.__b

    @classmethod
    def c (cls) -> np.ndarray:
        return cls.__c

    @classmethod
    def is_embedded_method (cls) -> bool:
        return False

class RungeKuttaFehlberg_4_5(RungeKutta_Explicit):
    """
    Runge-Kutta-Fehlberg 4(5) method.  This is a fourth-order RK method which uses a 5th order RK method
    to estimate the local truncation error.

    Reference:
    -   https://ipfs.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/List_of_Runge%E2%80%93Kutta_methods.html
    """

    # Define the Butcher tableau using class variables, so new np.ndarrays aren't created during the step function.
    __a = np.array([
        [   0.0   ,     0.0   ,     0.0   ,    0.0   ,   0.0 , 0.0],
        [   1/4   ,     0.0   ,     0.0   ,    0.0   ,   0.0 , 0.0],
        [   3/32  ,     9/32  ,     0.0   ,    0.0   ,   0.0 , 0.0],
        [1932/2197, -7200/2197,  7296/2197,    0.0   ,   0.0 , 0.0],
        [ 439/216 ,    -8.0   ,  3680/513 , -845/4104,   0.0 , 0.0],
        [  -8/27  ,     2.0   , -3544/2565, 1859/4104, -11/40, 0.0],
    ])
    __b = np.array([25/216, 0.0, 1408/2565, 2197/4104, -1/5, 0])
    __b_star = np.array([16/135, 0.0, 6656/12825, 28561/56430, -9/50, 2/55])
    __c = np.array([0.0, 1/4, 3/8, 12/13, 1.0, 1/2])

    @classmethod
    def order (cls) -> int:
        return 4

    @classmethod
    def a (cls) -> np.ndarray:
        return cls.__a

    @classmethod
    def b (cls) -> np.ndarray:
        return cls.__b

    @classmethod
    def b_star (cls) -> np.ndarray:
        return cls.__b_star

    @classmethod
    def c (cls) -> np.ndarray:
        return cls.__c

    @classmethod
    def is_embedded_method (cls) -> bool:
        return True

if __name__ == '__main__':
    def do_stuff_0 ():
        # Vector field of rigid counterclockwise rotation
        def V (t, y):
            return np.array([-y[1], y[0]])

        #integrator = RungeKutta_4(vector_field=V, parameter_shape=(2,))
        integrator = RungeKuttaFehlberg_4_5(vector_field=V, parameter_shape=(2,))
        t = 0.0
        y = np.array([1.0, 0.0])
        dt = 0.1
        t_max = 6.3

        t_v = [t]
        y_v = [np.copy(y)]
        ltee_v = [0.0]
        while t < t_max:
            integrator.set_inputs(t, y)
            integrator.step(dt)
            t, y = integrator.get_outputs()
            t_v.append(t)
            y_v.append(np.copy(y))
            ltee_v.append(np.sqrt(integrator.ltee_squared))

        print(f'ltee_v = {ltee_v}')

        # Convert the list of np.ndarray to a full np.ndarray.
        y_t = np.array(y_v)

        import matplotlib.pyplot as plt

        def plot_stuff ():
            row_count   = 1
            col_count   = 4
            size        = 5
            fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

            axis = axis_vv[0][0]
            axis.set_title('position')
            axis.set_aspect('equal')
            axis.plot(y_t[:,0], y_t[:,1], '.')

            axis = axis_vv[0][1]
            axis.set_title('x')
            axis.plot(t_v, y_t[:,0], '.')

            axis = axis_vv[0][2]
            axis.set_title('y')
            axis.plot(t_v, y_t[:,1], '.')

            axis = axis_vv[0][3]
            axis.set_title('local trunc. err. est.')
            axis.semilogy(t_v, ltee_v, '.')

            fig.tight_layout()
            filename = 'runge-kutta.png'
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

    def do_stuff_1 ():
        import sympy as sp
        import vorpy
        import vorpy.symbolic
        import vorpy.symplectic

        np.set_printoptions(precision=20)

        # Define the Kepler problem and use it to test the integrator

        def phase_space_coordinates ():
            return np.array(sp.var('x,y,p_x,p_y')).reshape(2,2)

        def K (p):
            return np.dot(p.flat, p.flat) / 2

        def U (q):
            return -1 / sp.sqrt(np.dot(q.flat, q.flat))

        def H (qp):
            """Total energy -- should be conserved."""
            return K(qp[1,...]) + U(qp[0,...])

        def p_theta (qp):
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
        qp_initial = np.array([[1.0,0.0],[0.0,0.5]])
        H_initial = H_fast(qp_initial)
        p_theta_initial = p_theta_fast(qp_initial)

        print(f'H_initial = {H_initial}')
        print(f'p_theta_initial = {p_theta_initial}')

        #integrator = RungeKutta_4(vector_field=V, parameter_shape=(2,))
        integrator = RungeKuttaFehlberg_4_5(vector_field=(lambda t,qp:X_H_fast(qp)), parameter_shape=vorpy.tensor.shape(qp_initial))
        t = t_initial
        y = qp_initial
        dt = 0.01
        t_max = 3.0

        t_v = [t]
        y_v = [np.copy(y)]
        ltee_v = [0.0]
        while t < t_max:
            integrator.set_inputs(t, y)
            integrator.step(dt)
            t, y = integrator.get_outputs()
            t_v.append(t)
            y_v.append(np.copy(y))
            ltee_v.append(np.sqrt(integrator.ltee_squared))

        print(f'ltee_v = {ltee_v}')

        # Convert the list of np.ndarray to a full np.ndarray.
        qp_t = np.array(y_v)

        H_v = vorpy.apply_along_axes(H_fast, (1,2), (qp_t,))
        H_error_v = vorpy.apply_along_axes(lambda qp:np.abs(H_fast(qp) - H_initial), (1,2), (qp_t,))
        #print(f'H_v = {H_v}')
        #print(f'H_error_v = {H_error_v}')

        p_theta_v = vorpy.apply_along_axes(p_theta_fast, (1,2), (qp_t,))
        p_theta_error_v = vorpy.apply_along_axes(lambda qp:np.abs(p_theta_fast(qp) - p_theta_initial), (1,2), (qp_t,))
        #print(f'p_theta_v = {p_theta_v}')
        #print(f'p_theta_error_v = {p_theta_error_v}')

        import matplotlib.pyplot as plt

        def plot_stuff ():
            row_count   = 1
            col_count   = 5
            size        = 5
            fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

            axis = axis_vv[0][0]
            axis.set_title('position')
            axis.set_aspect('equal')
            axis.plot(qp_t[:,0,0], qp_t[:,0,1], '.')

            axis = axis_vv[0][1]
            axis.set_title('x and y')
            axis.plot(t_v, qp_t[:,0,0], '.')
            axis.plot(t_v, qp_t[:,0,1], '.')

            axis = axis_vv[0][2]
            axis.set_title('local trunc. err. est.')
            axis.semilogy(t_v, ltee_v, '.')

            axis = axis_vv[0][3]
            axis.set_title('H error')
            axis.semilogy(t_v, H_error_v, '.')

            axis = axis_vv[0][4]
            axis.set_title('p_theta error')
            axis.semilogy(t_v, p_theta_error_v, '.')

            fig.tight_layout()
            filename = 'runge-kutta-kepler.png'
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

    #do_stuff_0()
    do_stuff_1()
