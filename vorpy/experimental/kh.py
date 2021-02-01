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

def arg (x:float, y:float, branch_center:float) -> float:
    # The bound is defined to be [branch_center-pi, branch_center+pi) (upper bound excluded).
    branch_bounds = (branch_center-np.pi, branch_center+np.pi)

    theta = np.arctan2(y, x)
    if theta < branch_center:
        while theta < branch_bounds[0]:
            theta += 2.0*np.pi
    else:
        while theta >= branch_bounds[1]:
            theta -= 2.0*np.pi

    assert branch_bounds[0] <= theta < branch_bounds[1]

def arg_v (x_v:np.ndarray, y_v:np.ndarray, branch_center_initial:float) -> float:
    if x_v.shape != y_v.shape:
        raise TypeError(f'expected x_v and y_v to have the same shape, but their shapes were {x_v.shape} and {y_v.shape} respectively')

    theta_v = np.ndarray(x_v.shape, dtype=x_v.dtype)
    theta_v = arg(x_v[0], y_v[0], branch_center_initial)
    for i in range(len(theta_v), 1):
        theta_v[i] = arg(x_v[i], y_v[i], theta_v[i-1])

    return theta_v

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
    def change_qp_coordinates_to (cls, other_coordinates:typing.Any, qp:np.ndarray) -> np.ndarray:
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

    #@classmethod
    #def Legendre_transform_pinv (cls, q:np.ndarray) -> np.ndarray:
        #LT = cls.Legendre_transform(q)
        #LT_pinv = sp.Matrix(LT).pinv()
        #LT_pinv.simplify()
        #LT_pinv = np.array(LT_pinv) # Make it back into a np.ndarray
        #return LT_pinv

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

    '''
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
    '''

    #@classmethod
    #def L__symbolic (cls, qv:np.ndarray) -> typing.Any:
        #"""
        #The Lagrangian and the Hamiltonian are related as follows.

            #H(q,p) + L(q,v) = p*v,

        #where p*v is the natural pairing of the cotangent-valued momentum (p)
        #and the tangent-valued velocity (v).
        #"""
        #q, v = qv
        #qp = cls.qv_to_qp__symbolic(qv)
        #p = qp[1,:]
        #return cls.H__symbolic(qp) - np.dot(p,v)

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
    def qp_coordinates (cls) -> np.ndarray:
        return np.array([
            [sp.var('x'),   sp.var('y'),   sp.var('z')],
            [sp.var('p_x'), sp.var('p_y'), sp.var('p_z')],
        ])

    @classmethod
    @abc.abstractmethod
    def change_qp_coordinates_to (cls, other_coordinates:typing.Any, qp:np.ndarray) -> np.ndarray:
        if qp.shape != (2,3):
            raise TypeError(f'expected qp.shape to be s+(2,3) for some shape s, but qp.shape was {qp.shape}')

        if other_coordinates is EuclideanSymbolics:
            return qp
        #elif other_coordinates is QuadraticCylindricalSymbolics:
            #qc_qp = np.ndarray(qp.shape, dtype=qp.dtype)

            #x   = qp[...,0,0]
            #y   = qp[...,0,1]
            #z   = qp[...,0,2]
            #p_x = qp[...,1,0]
            #p_y = qp[...,1,1]
            #p_z = qp[...,1,2]

            #R   = x**2 + y**2
            ## TODO: need to make continuous, moving choice of branch
            #theta =
        else:
            raise TypeError(f'coordinate change from {cls} to {other_coordinates} not implemented')

    @classmethod
    def qv_coordinates (cls) -> np.ndarray:
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
    def qp_coordinates (cls) -> np.ndarray:
        return np.array([
            [sp.var('R'),   sp.var('theta'),   sp.var('w')],
            [sp.var('p_R'), sp.var('p_theta'), sp.var('p_w')],
        ])

    @classmethod
    @abc.abstractmethod
    def change_qp_coordinates_to (cls, other_coordinates:typing.Any, qp:np.ndarray) -> np.ndarray:
        if qp.shape != (2,3):
            raise TypeError(f'expected qp.shape to be s+(2,3) for some shape s, but qp.shape was {qp.shape}')

        R       = qp[0,0]
        theta   = qp[0,1]
        w       = qp[0,2]
        p_R     = qp[1,0]
        p_theta = qp[1,1]
        p_w     = qp[1,2]

        if other_coordinates is EuclideanSymbolics:
            r       = sp.sqrt(R)

            x       = r*sp.cos(theta)
            y       = r*sp.sin(theta)
            z       = w/4
            p_x     = 2*r*p_R*sp.cos(theta) - p_theta*sp.sin(theta)/r
            p_y     = 2*r*p_R*sp.sin(theta) + p_theta*sp.cos(theta)/r
            p_z     = 4*p_w

            return np.array([
                [x,   y,   z],
                [p_x, p_y, p_z],
            ])
        elif other_coordinates is QuadraticCylindricalSymbolics:
            return qp
        elif other_coordinates is LogSizeSymbolics:
            s       = sp.log(R**2 + w**2) / 4
            u       = sp.atan2(w, R)
            p_s     = 2*R*p_R + 2*w*p_w
            p_u     = R*p_w - w*p_R

            return np.array([
                [s,   theta,   u],
                [p_s, p_theta, p_u],
            ])
        else:
            raise TypeError(f'coordinate change from {cls} to {other_coordinates} not implemented')

    @classmethod
    def qv_coordinates (cls) -> np.ndarray:
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

class LogSizeSymbolics(KeplerHeisenbergSymbolics):
    """
    Kepler-Heisenberg symbolic quantities in a log-size cylindrical coordinates (s, theta, u), where

        s     = log(R^2 + w^2) / 4
        theta = theta
        u     = arg(R, w)
    """

    @classmethod
    def name (cls) -> str:
        return 'LogSizeSymbolics'

    @classmethod
    def qp_coordinates (cls) -> np.ndarray:
        return np.array([
            [sp.var('s'),   sp.var('theta'),   sp.var('u')],
            [sp.var('p_s'), sp.var('p_theta'), sp.var('p_u')],
        ])

    @classmethod
    @abc.abstractmethod
    def change_qp_coordinates_to (cls, other_coordinates:typing.Any, qp:np.ndarray) -> np.ndarray:
        if qp.shape != (2,3):
            raise TypeError(f'expected qp.shape to be s+(2,3) for some shape s, but qp.shape was {qp.shape}')

        if other_coordinates is QuadraticCylindricalSymbolics:
            s       = qp[0,0]
            theta   = qp[0,1]
            u       = qp[0,2]
            p_s     = qp[1,0]
            p_theta = qp[1,1]
            p_u     = qp[1,2]

            R       = sp.exp(2*s)*sp.cos(u)
            w       = sp.exp(2*s)*sp.sin(u)
            p_R     = sp.exp(-2*s)*(sp.cos(u)*p_s/2 - sp.sin(u)*p_u)
            p_w     = sp.exp(-2*s)*(sp.sin(u)*p_s/2 + sp.cos(u)*p_u)

            return np.array([
                [R,   theta,   w],
                [p_R, p_theta, p_w],
            ])
        elif other_coordinates is LogSizeSymbolics:
            return qp
        elif other_coordinates is EuclideanSymbolics:
            s       = qp[0,0]
            theta   = qp[0,1]
            u       = qp[0,2]
            p_s     = qp[1,0]
            p_theta = qp[1,1]
            p_u     = qp[1,2]

            R       = sp.exp(2*s)*sp.cos(u)
            w       = sp.exp(2*s)*sp.sin(u)
            p_R     = sp.exp(-2*s)*(sp.cos(u)*p_s/2 - sp.sin(u)*p_u)
            p_w     = sp.exp(-2*s)*(sp.sin(u)*p_s/2 + sp.cos(u)*p_u)

            r       = sp.sqrt(R)

            x       = r*sp.cos(theta)
            y       = r*sp.sin(theta)
            z       = w/4
            p_x     = 2*r*p_R*sp.cos(theta) - p_theta*sp.sin(theta)/r
            p_y     = 2*r*p_R*sp.sin(theta) + p_theta*sp.cos(theta)/r
            p_z     = 4*w

            return np.array([
                [x,   y,   z],
                [p_x, p_y, p_z],
            ])
        else:
            raise TypeError(f'coordinate change from {cls} to {other_coordinates} not implemented')

    @classmethod
    def qv_coordinates (cls) -> np.ndarray:
        return np.array([
            [sp.var('s'),   sp.var('theta'),   sp.var('u')],
            [sp.var('v_s'), sp.var('v_theta'), sp.var('v_u')],
        ])

    @classmethod
    def H__symbolic (cls, qp:np.ndarray) -> typing.Any: # TODO: this should specify a scalar type somehow
        s,   theta,   u         = qp[0,:]
        p_s, p_theta, p_u   = p = qp[1,:]

        # (4*pi*p_theta**2 + 2*pi*(p_s**2*cos(2*u) + p_s**2 + 2*p_s*p_theta*sin(2*u) + 4*p_theta*p_u*cos(2*u) + 4*p_theta*p_u + 4*p_u**2*cos(2*u) + 4*p_u**2) - cos(u))*exp(-2*s)/(8*pi*cos(u))

        # M is related to the Legendre transform by a factor of exp(2*s).
        M                   = np.array([
            [sp.cos(u), sp.sin(u),   0],
            [sp.sin(u), 1/sp.cos(u), 2*sp.cos(u)],
            [0,         2*sp.cos(u), 4*sp.cos(u)],
        ])
        H                   = sp.exp(-2*s)*(vorpy.tensor.contract('i,ij,j', p, M, p, dtype=object)/2 - 1/(8*sp.pi))

        return H

    @classmethod
    def J__symbolic (cls, qp:np.ndarray) -> np.ndarray:
        p_s = qp[1,0]
        return p_s

    @classmethod
    def p_theta__symbolic (cls, qp:np.ndarray) -> np.ndarray:
        return qp[1,1]

    @staticmethod
    def p_u_constrained_by_H__symbolic (arguments:np.ndarray) -> np.ndarray:
        """
        Solves for p_u in terms of s, theta, u, p_s, p_theta, H_initial (which are the elements of the (6,)-shaped ndarray arguments).
        There are two solutions, and this returns them both as an np.ndarray with shape (2,).
        """

        if arguments.shape != (6,):
            raise TypeError(f'expected arguments.shape == (6,), but it was actually {arguments.shape}')

        # Unpack the arguments so they can form the specific expressions.
        s, theta, u, p_s, p_theta, H_initial    = arguments
        p_u                                     = sp.var('p_u')
        qp                                      = np.array([[s, theta, u], [p_s, p_theta, p_u]])
        H                                       = LogSizeSymbolics.H__symbolic(qp)

        p_u_constrained_by_H_v                  = sp.solve(H - H_initial, p_u)
        assert len(p_u_constrained_by_H_v) == 2
        return np.array(p_u_constrained_by_H_v)

    @classmethod
    @abc.abstractmethod
    def p_dependent_constrained_by_H__symbolic (cls, arguments:np.ndarray) -> np.ndarray:
        return LogSizeSymbolics.p_u_constrained_by_H__symbolic(arguments)

    @classmethod
    def p_s_bounds__symbolic (cls, u_p_theta:np.array) -> np.array:
        u, p_theta = u_p_theta

        x = -sp.tan(u)*p_theta
        discriminant = 2 * x**2 + 1/(4*sp.pi*sp.cos(u))
        return np.array([
            x - sp.sqrt(discriminant),
            x + sp.sqrt(discriminant),
        ])

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
    def generate_compute_trajectory_args (cls, base_dir_p:pathlib.Path): # TODO: generator return type
        pass

    @classmethod
    def compute_trajectory (cls, pickle_filename_p:pathlib.Path, qp_initial:np.ndarray, t_final:float, solution_sheet:int, return_y_jet:bool=False) -> vorpy.integration.adaptive.IntegrateVectorFieldResults:
        if qp_initial.shape != (2,3):
            raise TypeError(f'Expected qp_initial.shape == (2,3) but it was actually {qp_initial.shape}')

        H_initial = cls.H__fast(qp_initial) # type: ignore
        p_theta_initial = cls.p_theta__fast(qp_initial) # type: ignore

        H_cq = vorpy.integration.adaptive.ControlledQuantity(
            name='H',
            reference_quantity=H_initial,
            #global_error_band=vorpy.integration.adaptive.RealInterval(1.0e-12, 1.0e-10),
            global_error_band=vorpy.integration.adaptive.RealInterval(1.0e-12, 1.0e-6),
            quantity_evaluator=(lambda t,qp:typing.cast(float, cls.H__fast(qp))), # type: ignore
        )
        p_theta_cq = vorpy.integration.adaptive.ControlledQuantity(
            name='p_theta',
            reference_quantity=p_theta_initial,
            #global_error_band=vorpy.integration.adaptive.RealInterval(1.0e-12, 1.0e-10),
            global_error_band=vorpy.integration.adaptive.RealInterval(1.0e-12, 1.0e-6),
            quantity_evaluator=(lambda t,qp:typing.cast(float, cls.p_theta__fast(qp))), # type: ignore
        )
        controlled_sq_ltee = vorpy.integration.adaptive.ControlledSquaredLTEE(
            #global_error_band=vorpy.integration.adaptive.RealInterval(1.0e-14**2, 1.0e-10**2),
            global_error_band=vorpy.integration.adaptive.RealInterval(1.0e-14**2, 1.0e-6**2),
        )

        try:
            results = vorpy.integration.adaptive.integrate_vector_field(
                vector_field=(lambda t,qp:cls.X_H__fast(qp)), # type: ignore
                t_initial=0.0,
                y_initial=qp_initial,
                t_final=t_final,
                controlled_quantity_d={
                    'H abs error':H_cq,
                    'p_theta abs error':p_theta_cq,
                },
                controlled_sq_ltee=controlled_sq_ltee,
                return_y_jet=return_y_jet,
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
            raise

    @classmethod
    def compute_stuff (cls, base_dir_p:pathlib.Path) -> None:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            args_v = list(cls.generate_compute_trajectory_args(base_dir_p)) # type: ignore
            for args_index,result in enumerate(executor.map(cls.compute_trajectory__worker, args_v)): # type: ignore
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
    def H__fast () -> float:
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
    def qp_to_qv__fast () -> np.ndarray:
        qp = EuclideanSymbolics.qp_coordinates()
        return EuclideanSymbolics.qp_to_qv__symbolic(qp), qp

    #@staticmethod
    #@vorpy.symbolic.cache_lambdify(
        #function_id='Euclidean__qv_to_qp',
        #argument_id='qp',
        #replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt'},
        #import_v=['import numpy as np'],
        #verbose=True,
    #)
    #def qv_to_qp__fast () -> np.ndarray:
        #qv = EuclideanSymbolics.qv_coordinates()
        #return EuclideanSymbolics.qv_to_qp__symbolic(qv), qv

    #@staticmethod
    #@vorpy.symbolic.cache_lambdify(
        #function_id='Euclidean__L',
        #argument_id='qp',
        #replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt'},
        #import_v=['import numpy as np'],
        #verbose=True,
    #)
    #def L__fast () -> float:
        #qv = EuclideanSymbolics.qv_coordinates()
        #return EuclideanSymbolics.L__symbolic(qv), qv

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='Euclidean__X_H',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def X_H__fast () -> np.ndarray:
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
    def J__fast () -> float:
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
    def p_theta__fast () -> float:
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
    def p_z_constrained_by_H__fast () -> float:
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
    def qp_constrained_by_H__fast () -> np.ndarray:
        X = np.array(sp.var('x,y,z,p_x,p_y,H_initial'))
        return EuclideanSymbolics.qp_constrained_by_H__symbolic(X), X

    @classmethod
    def generate_compute_trajectory_args (cls, base_dir_p:pathlib.Path):
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
                pickle_filename_p = base_dir_p / cls.name() / f'H={H_initial}_J={J_initial}_sheet={solution_sheet}/trajectory-{trajectory_index:06}.pickle'
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
        function_id='QuadraticCylindrical__qp_to_Euclidean',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'cos':'np.cos', 'sin':'np.sin', 'sqrt':'np.sqrt', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def qp_to_Euclidean__fast () -> np.ndarray:
        qp = QuadraticCylindricalSymbolics.qp_coordinates()
        return QuadraticCylindricalSymbolics.change_qp_coordinates_to(EuclideanSymbolics, qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='QuadraticCylindrical__qp_to_LogSize',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'cos':'np.cos', 'sin':'np.sin', 'sqrt':'np.sqrt', 'log':'np.log', 'atan2':'np.arctan2', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def qp_to_LogSize__fast () -> np.ndarray:
        qp = QuadraticCylindricalSymbolics.qp_coordinates()
        return QuadraticCylindricalSymbolics.change_qp_coordinates_to(LogSizeSymbolics, qp), qp

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

    #@staticmethod
    #@vorpy.symbolic.cache_lambdify(
        #function_id='QuadraticCylindrical__qv_to_qp',
        #argument_id='qp',
        #replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt'},
        #import_v=['import numpy as np'],
        #verbose=True,
    #)
    #def qv_to_qp__fast ():
        #qv = QuadraticCylindricalSymbolics.qv_coordinates()
        #return QuadraticCylindricalSymbolics.qv_to_qp__symbolic(qv), qv

    #@staticmethod
    #@vorpy.symbolic.cache_lambdify(
        #function_id='QuadraticCylindrical__L',
        #argument_id='qp',
        #replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt'},
        #import_v=['import numpy as np'],
        #verbose=True,
    #)
    #def L__fast ():
        #qv = QuadraticCylindricalSymbolics.qv_coordinates()
        #return QuadraticCylindricalSymbolics.L__symbolic(qv), qv

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
    def generate_compute_trajectory_args (cls, base_dir_p:pathlib.Path):
        t_final = 60.0

        R_initial_v = [1.0]
        theta_initial_v = [0.0]
        w_initial_v = [0.0]

        p_R_initial_v = np.linspace(-0.125, 0.125, 21)
        assert 0.0 in p_R_initial_v

        p_theta_initial_v = np.linspace(0.05, 0.4, 31)

        #H_initial_v = np.linspace(-1.0/32, 1.0/32, 11)
        H_initial_v = [0.0]
        assert 0.0 in H_initial_v

        trajectory_index = 0

        for R_initial,theta_initial,w_initial,p_R_initial,p_theta_initial,H_initial in itertools.product(R_initial_v,theta_initial_v,w_initial_v,p_R_initial_v,p_theta_initial_v,H_initial_v):
            X = np.array([R_initial,theta_initial,w_initial,p_R_initial,p_theta_initial,H_initial])
            p_w_constrained_by_H_v = QuadraticCylindricalNumerics.p_w_constrained_by_H__fast(X)

            for solution_sheet,p_w_constrained_by_H in enumerate(p_w_constrained_by_H_v):
                qp_initial = np.array([[X[0], X[1], X[2]], [X[3], X[4], p_w_constrained_by_H]])
                J_initial = QuadraticCylindricalNumerics.J__fast(qp_initial)
                pickle_filename_p = base_dir_p / cls.name() / f'H={H_initial}_J={J_initial}_sheet={solution_sheet}/trajectory-{trajectory_index:06}.pickle'
                yield pickle_filename_p, qp_initial, t_final, solution_sheet

                trajectory_index += 1

    @staticmethod
    def compute_trajectory__worker (args):
        return QuadraticCylindricalNumerics.compute_trajectory(*args)

class LogSizeNumerics(KeplerHeisenbergNumerics):
    @classmethod
    def name (cls) -> str:
        return 'LogSizeNumerics'

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='LogSize__qp_to_QuadraticCylindrical',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'cos':'np.cos', 'sin':'np.sin', 'exp':'np.exp', 'pi':'np.pi', 'sqrt':'np.sqrt', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def qp_to_QuadraticCylindrical__fast () -> np.ndarray:
        qp = LogSizeSymbolics.qp_coordinates()
        return LogSizeSymbolics.change_qp_coordinates_to(QuadraticCylindricalSymbolics, qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='LogSize__qp_to_Euclidean',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'cos':'np.cos', 'sin':'np.sin', 'exp':'np.exp', 'pi':'np.pi', 'sqrt':'np.sqrt', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def qp_to_Euclidean__fast () -> np.ndarray:
        qp = LogSizeSymbolics.qp_coordinates()
        return LogSizeSymbolics.change_qp_coordinates_to(EuclideanSymbolics, qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='LogSize__H',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'exp':'np.exp', 'cos':'np.cos', 'sin':'np.sin', 'sqrt':'np.sqrt'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def H__fast ():
        qp = LogSizeSymbolics.qp_coordinates()
        return LogSizeSymbolics.H__symbolic(qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='LogSize__qp_to_qv',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'cos':'np.cos', 'sin':'np.sin', 'exp':'np.exp', 'sqrt':'np.sqrt'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def qp_to_qv__fast ():
        qp = LogSizeSymbolics.qp_coordinates()
        return LogSizeSymbolics.qp_to_qv__symbolic(qp), qp

    #@staticmethod
    #@vorpy.symbolic.cache_lambdify(
        #function_id='LogSize__qv_to_qp',
        #argument_id='qp',
        #replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt'},
        #import_v=['import numpy as np'],
        #verbose=True,
    #)
    #def qv_to_qp__fast ():
        #qv = LogSizeSymbolics.qv_coordinates()
        #return LogSizeSymbolics.qv_to_qp__symbolic(qv), qv

    #@staticmethod
    #@vorpy.symbolic.cache_lambdify(
        #function_id='LogSize__L',
        #argument_id='qp',
        #replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'sqrt':'np.sqrt'},
        #import_v=['import numpy as np'],
        #verbose=True,
    #)
    #def L__fast ():
        #qv = LogSizeSymbolics.qv_coordinates()
        #return LogSizeSymbolics.L__symbolic(qv), qv

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='LogSize__X_H',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'cos':'np.cos', 'sin':'np.sin', 'exp':'np.exp', 'sqrt':'np.sqrt', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def X_H__fast ():
        qp = LogSizeSymbolics.qp_coordinates()
        return LogSizeSymbolics.X_H__symbolic(qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='LogSize__J',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def J__fast ():
        qp = LogSizeSymbolics.qp_coordinates()
        return LogSizeSymbolics.J__symbolic(qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='LogSize__p_theta',
        argument_id='qp',
        replacement_d={'dtype=object':'dtype=float'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def p_theta__fast ():
        qp = LogSizeSymbolics.qp_coordinates()
        return LogSizeSymbolics.p_theta__symbolic(qp), qp

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='LogSize__p_u_constrained_by_H',
        argument_id='X',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'cos':'np.cos', 'sin':'np.sin', 'exp':'np.exp', 'sqrt':'np.sqrt', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def p_u_constrained_by_H__fast ():
        X = np.array(sp.var('s,theta,u,p_s,p_theta,H_initial'))
        return LogSizeSymbolics.p_u_constrained_by_H__symbolic(X), X

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='LogSize__qp_constrained_by_H',
        argument_id='X',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'cos':'np.cos', 'sin':'np.sin', 'exp':'np.exp', 'sqrt':'np.sqrt', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def qp_constrained_by_H__fast ():
        X = np.array(sp.var('s,theta,u,p_s,p_theta,H_initial'))
        return LogSizeSymbolics.qp_constrained_by_H__symbolic(X), X

    @staticmethod
    @vorpy.symbolic.cache_lambdify(
        function_id='LogSize__p_s_bounds',
        argument_id='X',
        replacement_d={'dtype=object':'dtype=float', 'pi':'np.pi', 'cos':'np.cos', 'sin':'np.sin', 'tan':'np.tan', 'exp':'np.exp', 'sqrt':'np.sqrt', 'ndarray':'np.ndarray'},
        import_v=['import numpy as np'],
        verbose=True,
    )
    def p_s_bounds__fast ():
        X = np.array(sp.var('u,p_theta'))
        return LogSizeSymbolics.p_s_bounds__symbolic(X), X

    @classmethod
    def generate_compute_trajectory_args (cls, base_dir_p:pathlib.Path):
        t_final = 60.0

        s_initial_v = [0.0]
        theta_initial_v = [0.0]
        u_initial_v = [0.0]

        p_s_initial_v = np.linspace(-0.125, 0.125, 21)
        assert 0.0 in p_s_initial_v

        p_theta_initial_v = np.linspace(0.05, 0.4, 31)

        #H_initial_v = np.linspace(-1.0/32, 1.0/32, 11)
        #assert 0.0 in H_initial_v
        H_initial_v = [0.0]

        trajectory_index = 0

        for s_initial,theta_initial,u_initial,p_s_initial,p_theta_initial,H_initial in itertools.product(s_initial_v,theta_initial_v,u_initial_v,p_s_initial_v,p_theta_initial_v,H_initial_v):
            X = np.array([s_initial,theta_initial,u_initial,p_s_initial,p_theta_initial,H_initial])
            p_u_constrained_by_H_v = LogSizeNumerics.p_u_constrained_by_H__fast(X)

            for solution_sheet,p_u_constrained_by_H in enumerate(p_u_constrained_by_H_v):
                qp_initial = np.array([[X[0], X[1], X[2]], [X[3], X[4], p_u_constrained_by_H]])
                pickle_filename_p = base_dir_p / cls.name() / f'H={H_initial}_p_s={p_s_initial}_p_theta={p_theta_initial}_sheet={solution_sheet}/trajectory-{trajectory_index:06}.pickle'
                yield pickle_filename_p, qp_initial, t_final, solution_sheet

                trajectory_index += 1

    @staticmethod
    def compute_trajectory__worker (args):
        return LogSizeNumerics.compute_trajectory(*args)

if __name__ == '__main__':
    #qp = EuclideanSymbolics.qp_coordinates()
    #X_H = EuclideanSymbolics.X_H__symbolic(qp)
    #qp = QuadraticCylindricalSymbolics.qp_coordinates()
    #X_H = QuadraticCylindricalSymbolics.X_H__symbolic(qp)
    qp = LogSizeSymbolics.qp_coordinates()
    X_H = LogSizeSymbolics.X_H__symbolic(qp)
    print(f'X_H:\n{X_H.reshape(-1,1)}')
