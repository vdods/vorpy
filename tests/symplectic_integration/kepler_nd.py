import numpy as np

class KeplerNd:
    """
    Defines the various geometric-mechanical structures of a Kepler system (stationary sun and moving planet)
    in arbitrary dimension.  The gravitational potential in R^3 is 1/r, but in general R^N the potential
    energy function is taken to be a fundamental solution (i.e. radially symmetric with some particular
    normalization) to the Laplace equation.  Therefore it is:

        V_1(r) = c_1*abs(r)     for N == 1,
        V_2(r) = c_2*log(r)     for N == 2,
        V_N(r) = -c_N*r^(2-N)   for N >= 3,

    where c_k is a positive, normalizing constant, which for example can be picked such that the flux of the solution
    through the unit sphere is 1.  In this class, each constant is assumed to be 1.

    Coordinates are assumed to have shape (2,N), i.e. np.array([q,p]), where q and p are the vector-valued
    pendular angle and momentum respectively.  The angle coordinates are assumed to be normal coordinates
    with origin denoting the downward, stable equilibrium position.  The potential energy function is the
    vertical position of the pendular mass.
    """

    @staticmethod
    def K (p):
        """Kinetic energy is a function of the momentum only.  It is assumed that the planet has unit mass."""
        return 0.5*np.sum(np.square(p))

    @staticmethod
    def V (q):
        """Potential energy is a function of the position only."""
        N = q.shape[0]
        r = np.linalg.norm(q)
        if N == 1:
            return np.abs(r)
        elif N == 2:
            return np.log(r)
        else:
            return -r**(2-N)

    @staticmethod
    def H (coordinates):
        """The Hamiltonian is the sum of kinetic and potential energy."""
        q = coordinates[0,:]
        p = coordinates[1,:]
        return KeplerNd.K(p) + KeplerNd.V(q)

    @staticmethod
    def dK_dp (p):
        return p

    @staticmethod
    def dV_dq (q):
        N = q.shape[0]
        if N == 1:
            return np.sign(q)
        elif N == 2:
            return q / np.sum(np.square(q))
        else:
            r_squared = np.sum(np.square(q))
            return (N-2) * r_squared**((1-N)/2) * q

    @staticmethod
    def dH_dq (q, p):
        return KeplerNd.dV_dq(q)

    @staticmethod
    def dH_dp (q, p):
        return p

    @staticmethod
    def X_H (coordinates, *args): # args is assumed to be the time coordinate and other ignored args
        q = coordinates[0,:]
        p = coordinates[1,:]
        # This is the symplectic gradient of H.
        return np.array((KeplerNd.dK_dp(p), -KeplerNd.dV_dq(q)))

