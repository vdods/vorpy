import numpy as np
import vorpy.symbolic

def symplectic_dual_of_vector_field (V):
    """
    V must have shape (2,n) for some n > 0, and must be in Darboux coordinates.
    In particular, this implies that the symplectic form and its inverse are
    respectively written

        [ 0 -I ]        [  0 I ]
        [ I  0 ]        [ -I 0 ]

    The symplectic dual of a vector field is the symplectic form applied to
    the vector field.
    """
    if len(V.shape) != 2 or V.shape[0] != 2 or V.shape[1] == 0:
        raise FancyException(f'Expected V to have shape (2,n) for some n > 0, but instead it had shape {V.shape}')

    return np.vstack((-V[1,:], V[0,:]))

def symplectic_dual_of_covector_field (C):
    """
    C must have shape (2,n) for some n > 0, and must be in Darboux coordinates.
    In particular, this implies that the symplectic form and its inverse are
    respectively written

        [ 0 -I ]        [  0 I ]
        [ I  0 ]        [ -I 0 ]

    The symplectic dual of a covector field is the inverse of the symplectic form applied to
    the covector field.
    """
    if len(C.shape) != 2 or C.shape[0] != 2 or C.shape[1] == 0:
        raise FancyException(f'Expected C to have shape (2,n) for some n > 0, but instead it had shape {C.shape}')

    return np.vstack((C[1,:], -C[0,:]))

def symplectic_gradient_of (f, X):
    """
    Returns the symplectic dual of the covector field df.

    There is a choice of convention regarding where a particular negative sign goes (which
    arguably stems from the sign in the tautological 1-form on the cotangent bundle).  This
    function is defined such that the flow equation for the resulting vector field is
    Hamilton's equations of motion for the Hamiltonian function f.
    """

    #df_dq = vorpy.symbolic.differential(f, X[0,:])
    #df_dp = vorpy.symbolic.differential(f, X[1,:])
    #return np.vstack((df_dp, -df_dq))
    df = vorpy.symbolic.differential(f, X)
    assert df.shape == (2,3)
    return symplectic_dual_of_covector_field(df)
