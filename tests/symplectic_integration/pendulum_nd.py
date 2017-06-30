import numpy as np

class PendulumNd:
    """
    Defines the various geometric-mechanical structures of a spherical pendulum in arbitrary dimension.

    Coordinates are assumed to have shape (2,N), i.e. np.array([q,p]), where q and p are the vector-valued
    pendular angle and momentum respectively.  The angle coordinates are assumed to be normal coordinates
    with origin denoting the downward, stable equilibrium position.  The potential energy function is the
    vertical position of the pendular mass.
    """

    @staticmethod
    def K (p):
        """Kinetic energy is a function of the momentum only.  It is assumed that the pendulum has unit mass."""
        return 0.5*np.sum(np.square(p))

    @staticmethod
    def V (q):
        """Potential energy is a function of the position only."""
        # np.linalg.norm(q) gives the angle from the vertical axis
        return -np.cos(np.linalg.norm(q))

    @staticmethod
    def H (coordinates):
        """The Hamiltonian is the sum of kinetic and potential energy."""
        q = coordinates[0,:]
        p = coordinates[1,:]
        return PendulumNd.K(p) + PendulumNd.V(q)

    @staticmethod
    def dK_dp (p):
        return p

    @staticmethod
    def dV_dq (q):
        # sinc(x) is sin(pi*x)/(pi*x) when x is not 0 -- this is used to avoid the singularity in sin(x)/x.
        return -np.sinc(np.linalg.norm(q)/np.pi) * q

    @staticmethod
    def dH_dq (q, p):
        return PendulumNd.dV_dq(q)

    @staticmethod
    def dH_dp (q, p):
        return p

    @staticmethod
    def X_H (coordinates, *args): # args is assumed to be the time coordinate and other ignored args
        q = coordinates[0,:]
        p = coordinates[1,:]
        # This is the symplectic gradient of H.
        return np.array((PendulumNd.dK_dp(p), -PendulumNd.dV_dq(q)))

