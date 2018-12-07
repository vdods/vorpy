import abc
import concurrent.futures
import itertools
import numpy as np
import pathlib
import sympy as sp
import sys
import typing
import vorpy.integration.adaptive
import vorpy.pickle
import vorpy.symplectic

class KeplerHeisenbergSymbolics:
    """
    Base class representing the symbolic quantities in the Kepler-Heisenberg problem.
    Subclasses give coordinate-chart-specific expressions for the various quantities.

    TODO: Maybe rename this to conform to the coordinate chart / atlas pattern.
    TODO: Make change-of-coordinate maps
    """

    @classmethod
    @abc.abstractmethod
    def name (cls) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def qp_coordinates (cls) -> np.ndarray:
        """Create symbolic Darboux coordinates for phase space (i.e. q=position, p=momentum)."""
        pass

    @classmethod
    @abc.abstractmethod
    def qv_coordinates (cls) -> np.ndarray:
        """Create coordinates for the tangent bundle of configuration space (i.e. q=position, v=velocity)."""
        pass

    @classmethod
    @abc.abstractmethod
    def H__symbolic (cls, qp:np.ndarray) -> typing.Any:
        pass

    @classmethod
    def Legendre_transform (cls, q:np.ndarray) -> np.ndarray:
        """
        The Legendre transform maps momenta into velocities.  In particular, the kinetic
        energy is assumed to have the form

            K(p) = p^T*LT*p / 2

        for some symmetric matrix LT, which is the coordinate expression for the Legendre transform.
        Assuming that the potential energy function doesn't depend on p, then by Hamilton's equations,

            v = dq/dt = dH/dp = dK/dp = p^T*LT.

        The matrix LT is just the Hessian of H with respect to p (again, assuming that the potential
        energy function doesn't depend on p).

        References:
        -   https://physics.stackexchange.com/questions/4384/physical-meaning-of-legendre-transformation
        """
        qp = cls.qp_coordinates()
        qp[0,:] = q
        p = qp[1,:]
        H = cls.H__symbolic(qp)
        LT = vorpy.symbolic.differential(vorpy.symbolic.differential(H, p), p)
        print(f'LT:\n{LT}')
        assert np.all(vorpy.symbolic.differential(LT, p) == 0), 'sanity check that H is actually quadratic in p, and hence LT does not depend on p'

        x,y,z = q

        eig0 = np.array([y/2, -x/2, 1])
        print(f'LT*eig0 = {np.dot(LT, eig0)}')

        eig1 = np.array([-y/x, 1, (x**2 + y**2)/(2*x)])
        print(f'LT*eig1 = {np.dot(LT, eig1)}')

        #eig2 = np.array([-y/x, 1, (x**2 + y**2)/(2*x)])
        #print(f'LT*eig1 = {np.dot(LT, eig1) - eig1}')

        det = sp.Matrix(LT).det().simplify()
        print(f'det(LT) = {det}')
        det = sp.Matrix(LT - np.eye(3, dtype=sp.Integer)).det().simplify()
        print(f'det(LT - I) = {det}')
        det = sp.Matrix(LT - np.eye(3, dtype=sp.Integer)*(1 + (x**2 + y**2)/4)).det().simplify()
        print(f'det(LT - (1+r**2/4)*I) = {det}')
        #eig1 =

        return LT

    @classmethod
    def Legendre_transform_pinv (cls, q:np.ndarray) -> np.ndarray:
        LT = cls.Legendre_transform(q)
        LT_pinv = sp.Matrix(LT).pinv()
        LT_pinv.simplify()
        LT_pinv = np.array(LT_pinv) # Make it back into a np.ndarray
        return LT_pinv

    @classmethod
    def qp_to_qv__symbolic (cls, qp:np.ndarray) -> np.ndarray:
        """
        The Legendre transform relates (q,p) to (q,v) (i.e. position+momentum to position+velocity).

        This can be seen from Hamilton's equations.

            dq/dt =  dH/dp
            dp/dt = -dH/dq,

        and since v := dq/dt, it follows that v = dH/dp.
        """
        q, p = qp
        LT = cls.Legendre_transform(q)
        print(f'LT:\n{LT}')
        print(f'p:\n{p}')
        det = sp.Matrix(LT).det()
        print(f'det = {det}')
        return np.vstack((q, np.dot(LT,p)))

    @classmethod
    def qv_to_qp__symbolic (cls, qv:np.ndarray) -> np.ndarray:
        """
        The Legendre transform relates (q,p) to (q,v) (i.e. position+momentum to position+velocity).

        This can be seen from Hamilton's equations.

            dq/dt =  dH/dp
            dp/dt = -dH/dq,

        and since v := dq/dt, it follows that v = dH/dp.
        """
        q, v = qv
        LT_pinv = cls.Legendre_transform_pinv(q)
        return np.vstack((q, np.dot(LT_pinv,v)))

        ## Create qp coordinates, but overwrite the q portion with qv's q portion, so that they're the same q.
        #qp = cls.qp_coordinates()
        #qp[0,:] = q
        #p = qp[1,:]
        ## Invert the transformation for qp_to_qv__symbolic.
        #H = cls.H__symbolic(qp)
        #dH_dp = vorpy.symbolic.differential(H, p)
        ##K_quadform = sp.Matrix(vorpy.symbolic.differential(vorpy.symbolic.differential(H, p), p))

        #dH_dp[...] = np.vectorize(sp.expand)(dH_dp)
        #print(f'dH_dp = {dH_dp}')
        #print(f'type(dH_dp) = {type(dH_dp)}')
        #equations = np.vectorize(lambda expr:sp.collect(expr, p))(dH_dp - v)
        #print(f'equations = {equations}')
        #p_solution_v = sp.linsolve((dH_dp - v).tolist(), p.tolist())
        #print(f'p_solution_v = {p_solution_v}')
        #assert len(p_solution_v) == 1
        #p_solution = p_solution_v[0]
        #return np.vstack((q, p_solution))

    @classmethod
    def L__symbolic (cls, qv:np.ndarray) -> typing.Any:
        """
        The Lagrangian and the Hamiltonian are related as follows.

            H(q,p) + L(q,v) = p*v,

        where p*v is the natural pairing of the cotangent-valued momentum (p)
        and the tangent-valued velocity (v).
        """
        q, v = qv
        qp = cls.qv_to_qp__symbolic(qv)
        p = qp[1,:]
        return H__symbolic(qp) - np.dot(p,v)

    @classmethod
    def X_H__symbolic (cls, qp:np.ndarray) -> np.ndarray:
        return vorpy.symplectic.symplectic_gradient_of(cls.H__symbolic(qp), qp)

    @classmethod
    @abc.abstractmethod
    def J__symbolic (cls, qp:np.ndarray) -> typing.Any:
        pass

    @classmethod
    @abc.abstractmethod
    def p_theta__symbolic (cls, qp:np.ndarray) -> typing.Any:
        pass

    @classmethod
    @abc.abstractmethod
    def p_dependent_constrained_by_H__symbolic (cls, arguments:np.ndarray) -> np.ndarray:
        """
        Arguments are [a_0, a_1, a_2, a_3, a_4, H_initial], where the a_i are coordinate-chart
        dependent -- all but the [1,2] element of the phase space coordinates.  H_initial
        specifies the fixed value of H to be used to solve for the remaining coordinate, which
        here is denoted p_dependent.  In particular, p_dependent is p_z for Euclidean, and is
        p_w for QuadraticCylindrical.

        If p_dependent is a solution (there may be more than one), then

            cls.H__symbolic(np.array([[a_0, a_1, a_2], [a_3, a_4, p_dependent]])) == H_initial

        Returns a numpy.array of shape (n,), containing all solutions to p_dependent, one per
        sheet of the solution.
        """
        pass

    @classmethod
    def qp_constrained_by_H__symbolic (cls, arguments:np.ndarray) -> np.ndarray:
        """
        Returns a numpy.array of shape (n,2,3), where the first index specifies which
        sheet the solution is drawn from.
        """

        p_dependent_v               = cls.p_dependent_constrained_by_H__symbolic(arguments)
        assert len(np.shape(p_dependent_v)) == 1
        n                           = len(p_dependent_v)
        H_initial                   = arguments[5]

        retval                      = np.ndarray((n,2,3), dtype=object)
        retval.reshape(n,-1)[:,0:5] = arguments[np.newaxis,0:5]
        retval[:,1,2]               = p_dependent_v

        assert np.all(np.array(sp.simplify(vorpy.apply_along_axes(cls.H__symbolic, (1,2), (retval,)) - H_initial)) == 0)

        return retval

class EuclideanSymbolics(KeplerHeisenbergSymbolics):
    """Kepler-Heisenberg symbolic quantities in Euclidean coordinates."""

    @classmethod
    def name (cls) -> str:
        return 'EuclideanSymbolics'

    @classmethod
    def qp_coordinates (cls):
        return np.array([
            [sp.var('x'),   sp.var('y'),   sp.var('z')],
            [sp.var('p_x'), sp.var('p_y'), sp.var('p_z')],
        ])

    @classmethod
    def qv_coordinates (cls):
        return np.array([
            [sp.var('x'),   sp.var('y'),   sp.var('z')],
            [sp.var('v_x'), sp.var('v_y'), sp.var('v_z')],
        ])

    @classmethod
    def H__symbolic (cls, qp:np.ndarray) -> typing.Any: # TODO: this should specify a scalar type somehow
        x, y, z         = qp[0,:]
        p_x, p_y, p_z   = qp[1,:]

        P_x             = p_x - y*p_z/2
        P_y             = p_y + x*p_z/2
        R               = x**2 + y**2
        H               = (P_x**2 + P_y**2)/2 - 1/(8*sp.pi*sp.sqrt(R**2 + 16*z**2))

        return H

    @classmethod
    def J__symbolic (cls, qp:np.ndarray) -> np.ndarray:
        x, y, z         = qp[0,:]
        p_x, p_y, p_z   = qp[1,:]

        return x*p_x + y*p_y + 2*z*p_z

    @classmethod
    def p_theta__symbolic (cls, qp:np.ndarray) -> np.ndarray:
        x, y, _         = qp[0,:]
        p_x, p_y, _     = qp[1,:]

        return x*p_y - y*p_x

    @staticmethod
    def p_z_constrained_by_H__symbolic (arguments:np.ndarray) -> np.ndarray:
        """
        Solves for p_z in terms of x, y, z, p_x, p_y, H_initial (which are the elements of the (6,)-shaped ndarray arguments).
        There are two solutions, and this returns them both as an np.ndarray with shape (2,).
        """

        if arguments.shape != (6,):
            raise TypeError(f'expected arguments.shape == (6,), but it was actually {arguments.shape}')

        # Unpack the arguments so they can form the specific expressions.
        x, y, z, p_x, p_y, H_initial    = arguments
        p_z                             = sp.var('p_z')
        qp                              = np.array([[x, y, z], [p_x, p_y, p_z]])
        H                               = EuclideanSymbolics.H__symbolic(qp)

        p_z_constrained_by_H_v          = sp.solve(H - H_initial, p_z)
        assert len(p_z_constrained_by_H_v) == 2
        return np.array(p_z_constrained_by_H_v)

    @classmethod
    @abc.abstractmethod
    def p_dependent_constrained_by_H__symbolic (cls, arguments:np.ndarray) -> np.ndarray:
        return EuclideanSymbolics.p_z_constrained_by_H__symbolic(arguments)

class QuadraticCylindricalSymbolics(KeplerHeisenbergSymbolics):
    """
    Kepler-Heisenberg symbolic quantities in a modified cylindrical coordinates (R, theta, w), where

        R     = x^2 + y^2
        theta = arg(x, y)
        w     = 4*z
    """

    @classmethod
    def name (cls) -> str:
        return 'QuadraticCylindricalSymbolics'

    @classmethod
    def qp_coordinates (cls):
        return np.array([
            [sp.var('R'),   sp.var('theta'),   sp.var('w')],
            [sp.var('p_R'), sp.var('p_theta'), sp.var('p_w')],
        ])

    @classmethod
    def qv_coordinates (cls):
        return np.array([
            [sp.var('R'),   sp.var('theta'),   sp.var('w')],
            [sp.var('v_R'), sp.var('v_theta'), sp.var('v_w')],
        ])

    @classmethod
    def H__symbolic (cls, qp:np.ndarray) -> typing.Any: # TODO: this should specify a scalar type somehow
        R,   theta,   w     = qp[0,:]
        p_R, p_theta, p_w   = qp[1,:]

        r                   = sp.sqrt(R)
        P_R                 = 2*r*p_R
        P_theta             = p_theta/r + 2*r*p_w
        rho                 = sp.sqrt(R**2 + w**2)
        H                   = (P_R**2 + P_theta**2)/2 - 1/(8*sp.pi*rho)

        return H

    @classmethod
    def J__symbolic (cls, qp:np.ndarray) -> np.ndarray:
        R,   theta,   w     = qp[0,:]
        p_R, p_theta, p_w   = qp[1,:]

        return 2*(R*p_R + w*p_w)

    @classmethod
    def p_theta__symbolic (cls, qp:np.ndarray) -> np.ndarray:
        return qp[1,1]

    @staticmethod
    def p_w_constrained_by_H__symbolic (arguments:np.ndarray) -> np.ndarray:
        """
        Solves for p_w in terms of R, theta, w, p_R, p_theta, H_initial (which are the elements of the (6,)-shaped ndarray arguments).
        There are two solutions, and this returns them both as an np.ndarray with shape (2,).
        """

        if arguments.shape != (6,):
            raise TypeError(f'expected arguments.shape == (6,), but it was actually {arguments.shape}')

        # Unpack the arguments so they can form the specific expressions.
        R, theta, w, p_R, p_theta, H_initial    = arguments
        p_w                                     = sp.var('p_w')
        qp                                      = np.array([[R, theta, w], [p_R, p_theta, p_w]])
        H                                       = QuadraticCylindricalSymbolics.H__symbolic(qp)

        p_w_constrained_by_H_v                  = sp.solve(H - H_initial, p_w)
        assert len(p_w_constrained_by_H_v) == 2
        return np.array(p_w_constrained_by_H_v)

    @classmethod
    @abc.abstractmethod
    def p_dependent_constrained_by_H__symbolic (cls, arguments:np.ndarray) -> np.ndarray:
        return QuadraticCylindricalSymbolics.p_w_constrained_by_H__symbolic(arguments)

class KeplerHeisenbergNumerics:
    """
    Base class representing the numeric quantities in the Kepler-Heisenberg problem.
    Subclasses give coordinate-chart-specific expressions for the various quantities.

    TODO: Maybe rename this to conform to the coordinate chart / atlas pattern.
    TODO: Make change-of-coordinate maps
    """

    @classmethod
    @abc.abstractmethod
    def name (cls) -> str:
        pass

    # TODO: Try to make a cached-lambdified classmethod.  if this is possible, then this
    # would greatly simplify a lot of things (could move the __fast methods into this baseclass.

    @classmethod
    @abc.abstractmethod
    def generate_compute_trajectory_args (cls):
        pass

    @classmethod
    def compute_trajectory (cls, pickle_filename_p:pathlib.Path, qp_initial:np.ndarray, t_final:float, solution_sheet:int) -> None:
        if qp_initial.shape != (2,3):
            raise TypeError(f'Expected qp_initial.shape == (2,3) but it was actually {qp_initial.shape}')

        H_initial = cls.H__fast(qp_initial)
        p_theta_initial = cls.p_theta__fast(qp_initial)

        H_cq = vorpy.integration.adaptive.ControlledQuantity(
            name='H',
            reference_quantity=H_initial,
            global_error_band=vorpy.integration.adaptive.RealInterval(1.0e-12, 1.0e-10),
            quantity_evaluator=(lambda t,qp:typing.cast(float, cls.H__fast(qp))),
        )
        p_theta_cq = vorpy.integration.adaptive.ControlledQuantity(
            name='p_theta',
            reference_quantity=p_theta_initial,
            global_error_band=vorpy.integration.adaptive.RealInterval(1.0e-12, 1.0e-10),
            quantity_evaluator=(lambda t,qp:typing.cast(float, cls.p_theta__fast(qp))),
        )
        controlled_sq_ltee = vorpy.integration.adaptive.ControlledSquaredLTEE(
            global_error_band=vorpy.integration.adaptive.RealInterval(1.0e-14**2, 1.0e-10**2),
        )

        try:
            results = vorpy.integration.adaptive.integrate_vector_field(
                vector_field=(lambda t,qp:cls.X_H__fast(qp)),
                t_initial=0.0,
                y_initial=qp_initial,
                t_final=t_final,
                controlled_quantity_d={
                    'H abs error':H_cq,
                    'p_theta abs error':p_theta_cq,
                },
                controlled_sq_ltee=controlled_sq_ltee,
            )

            pickle_filename_p.parent.mkdir(parents=True, exist_ok=True)

            data_d = dict(
                coordinates_name=cls.name(),
                solution_sheet=solution_sheet,
                results=results,
            )
            vorpy.pickle.pickle(data=data_d, pickle_filename=str(pickle_filename_p), log_out=sys.stdout)

            return results
        except ValueError as e:
            print(f'Caught exception {e} for qp_initial = {qp_initial}; pickle_filename_p = "{pickle_filename_p}"')

    @classmethod
    def compute_stuff (cls) -> None:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            args_v = list(cls.generate_compute_trajectory_args())
            for args_index,result in enumerate(executor.map(cls.compute_trajectory__worker, args_v)):
                print(f'{100*(args_index+1)//len(args_v): 3}% complete')

class EuclideanNumerics(KeplerHeisenbergNumerics):
    @classmethod
    def name (cls) -> str:
        return 'EuclideanNumerics'

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='Euclidean__H',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def H__fast ():
        qp = EuclideanSymbolics.qp_coordinates()
        return EuclideanSymbolics.H__symbolic(qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='Euclidean__qp_to_qv',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def qp_to_qv__fast ():
        qp = EuclideanSymbolics.qp_coordinates()
        return EuclideanSymbolics.qp_to_qv__symbolic(qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='Euclidean__qv_to_qp',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def qv_to_qp__fast ():
        qv = EuclideanSymbolics.qv_coordinates()
        return EuclideanSymbolics.qv_to_qp__symbolic(qv), qv

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='Euclidean__L',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def L__fast ():
        qv = EuclideanSymbolics.qv_coordinates()
        return EuclideanSymbolics.L__symbolic(qv), qv

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='Euclidean__X_H',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def X_H__fast ():
        qp = EuclideanSymbolics.qp_coordinates()
        return EuclideanSymbolics.X_H__symbolic(qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='Euclidean__J',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def J__fast ():
        qp = EuclideanSymbolics.qp_coordinates()
        return EuclideanSymbolics.J__symbolic(qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='Euclidean__p_theta',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def p_theta__fast ():
        qp = EuclideanSymbolics.qp_coordinates()
        return EuclideanSymbolics.p_theta__symbolic(qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='Euclidean__p_z_constrained_by_H',
        argument_id='X',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def p_z_constrained_by_H__fast ():
        X = np.array(sp.var('x,y,z,p_x,p_y,H_initial'))
        return EuclideanSymbolics.p_z_constrained_by_H__symbolic(X), X

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='Euclidean__qp_constrained_by_H',
        argument_id='X',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def qp_constrained_by_H__fast ():
        X = np.array(sp.var('x,y,z,p_x,p_y,H_initial'))
        return EuclideanSymbolics.qp_constrained_by_H__symbolic(X), X

    @classmethod
    def generate_compute_trajectory_args (cls):
        t_final = 60.0

        x_initial_v = [1.0]
        y_initial_v = [0.0]
        z_initial_v = [0.0]

        p_x_initial_v = np.linspace(-0.25, 0.25, 11)
        assert 0.0 in p_x_initial_v

        p_y_initial_v = np.linspace(0.05, 0.4, 11)

        H_initial_v = [-1.0/32, 0.0, 1.0/32]

        trajectory_index = 0

        for x_initial,y_initial,z_initial,p_x_initial,p_y_initial,H_initial in itertools.product(x_initial_v,y_initial_v,z_initial_v,p_x_initial_v,p_y_initial_v,H_initial_v):
            X = np.array([x_initial,y_initial,z_initial,p_x_initial,p_y_initial,H_initial])
            p_z_constrained_by_H_v = EuclideanNumerics.p_z_constrained_by_H__fast(X)

            for solution_sheet,p_z_constrained_by_H in enumerate(p_z_constrained_by_H_v):
                qp_initial = np.array([[X[0], X[1], X[2]], [X[3], X[4], p_z_constrained_by_H]])
                J_initial = EuclideanNumerics.J__fast(qp_initial)
                pickle_filename_p = pathlib.Path(f'kh_dilation_data.01/{cls.name()}/H={H_initial}_J={J_initial}_sheet={solution_sheet}/trajectory-{trajectory_index:06}.pickle')
                yield pickle_filename_p, qp_initial, t_final, solution_sheet

                trajectory_index += 1

    @staticmethod
    def compute_trajectory__worker (args):
        return EuclideanNumerics.compute_trajectory(*args)

class QuadraticCylindricalNumerics(KeplerHeisenbergNumerics):
    @classmethod
    def name (cls) -> str:
        return 'QuadraticCylindricalNumerics'

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='QuadraticCylindrical__H',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def H__fast ():
        qp = QuadraticCylindricalSymbolics.qp_coordinates()
        return QuadraticCylindricalSymbolics.H__symbolic(qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='QuadraticCylindrical__qp_to_qv',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def qp_to_qv__fast ():
        qp = QuadraticCylindricalSymbolics.qp_coordinates()
        return QuadraticCylindricalSymbolics.qp_to_qv__symbolic(qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='QuadraticCylindrical__qv_to_qp',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def qv_to_qp__fast ():
        qv = QuadraticCylindricalSymbolics.qv_coordinates()
        return QuadraticCylindricalSymbolics.qv_to_qp__symbolic(qv), qv

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='QuadraticCylindrical__L',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def L__fast ():
        qv = QuadraticCylindricalSymbolics.qv_coordinates()
        return QuadraticCylindricalSymbolics.L__symbolic(qv), qv

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='QuadraticCylindrical__X_H',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def X_H__fast ():
        qp = QuadraticCylindricalSymbolics.qp_coordinates()
        return QuadraticCylindricalSymbolics.X_H__symbolic(qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='QuadraticCylindrical__J',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def J__fast ():
        qp = QuadraticCylindricalSymbolics.qp_coordinates()
        return QuadraticCylindricalSymbolics.J__symbolic(qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='QuadraticCylindrical__p_theta',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def p_theta__fast ():
        qp = QuadraticCylindricalSymbolics.qp_coordinates()
        return QuadraticCylindricalSymbolics.p_theta__symbolic(qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='QuadraticCylindrical__p_w_constrained_by_H',
        argument_id='X',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def p_w_constrained_by_H__fast ():
        X = np.array(sp.var('R,theta,w,p_R,p_theta,H_initial'))
        return QuadraticCylindricalSymbolics.p_w_constrained_by_H__symbolic(X), X

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='QuadraticCylindrical__qp_constrained_by_H',
        argument_id='X',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def qp_constrained_by_H__fast ():
        X = np.array(sp.var('R,theta,w,p_R,p_theta,H_initial'))
        return QuadraticCylindricalSymbolics.qp_constrained_by_H__symbolic(X), X

    @classmethod
    def generate_compute_trajectory_args (cls):
        t_final = 60.0

        R_initial_v = [1.0]
        theta_initial_v = [0.0]
        w_initial_v = [0.0]

        p_R_initial_v = np.linspace(-0.125, 0.125, 21)
        assert 0.0 in p_R_initial_v

        p_theta_initial_v = np.linspace(0.05, 0.4, 31)

        H_initial_v = np.linspace(-1.0/32, 1.0/32, 11)
        assert 0.0 in H_initial_v

        trajectory_index = 0

        for R_initial,theta_initial,w_initial,p_R_initial,p_theta_initial,H_initial in itertools.product(R_initial_v,theta_initial_v,w_initial_v,p_R_initial_v,p_theta_initial_v,H_initial_v):
            X = np.array([R_initial,theta_initial,w_initial,p_R_initial,p_theta_initial,H_initial])
            p_w_constrained_by_H_v = QuadraticCylindricalNumerics.p_w_constrained_by_H__fast(X)

            for solution_sheet,p_w_constrained_by_H in enumerate(p_w_constrained_by_H_v):
                qp_initial = np.array([[X[0], X[1], X[2]], [X[3], X[4], p_w_constrained_by_H]])
                J_initial = QuadraticCylindricalNumerics.J__fast(qp_initial)
                pickle_filename_p = pathlib.Path(f'kh_dilation_data.01/{cls.name()}/H={H_initial}_J={J_initial}_sheet={solution_sheet}/trajectory-{trajectory_index:06}.pickle')
                yield pickle_filename_p, qp_initial, t_final, solution_sheet

                trajectory_index += 1

    @staticmethod
    def compute_trajectory__worker (args):
        return QuadraticCylindricalNumerics.compute_trajectory(*args)

import matplotlib.pyplot as plt
import scipy.interpolate
import vorpy.realfunction.piecewiselinear

def plot_J_equal_zero_extrapolated_trajectory (p_y_initial:float) -> None:
    x_initial = 1.0
    y_initial = 0.0
    z_initial = 0.0
    p_x_initial = 0.0
    H_initial = 0.0
    solution_sheet = 0

    qp_initial = EuclideanNumerics.qp_constrained_by_H__fast(
        np.array([x_initial, y_initial, z_initial, p_x_initial, p_y_initial, H_initial])
    )[solution_sheet]

    pickle_file_p = pathlib.Path(f'kh_dilation/qp.p_y={p_y_initial}.pickle')
    pickle_file_p.parent.mkdir(parents=True, exist_ok=True)

    results = EuclideanNumerics.compute_trajectory(
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


    plot_p = pathlib.Path(f'kh_dilation/qp.p_y={p_y_initial}.png')

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

    qp_initial = EuclideanNumerics.qp_constrained_by_H__fast(
        np.array([x_initial, y_initial, z_initial, p_x_initial, p_y_initial, H_initial])
    )[solution_sheet]

    pickle_file_p = pathlib.Path(f'kh_dilation/qp.p_y={p_y_initial}.pickle')
    pickle_file_p.parent.mkdir(parents=True, exist_ok=True)

    results = EuclideanNumerics.compute_trajectory(
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


    plot_p = pathlib.Path(f'kh_dilation/qp.p_y={p_y_initial}.png')

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
    pass
