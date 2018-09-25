import numpy as np
import vorpy.symbolic

def apply_vector_field_to_function (V, f, X):
    """This returns the directional derivative of f along V in coordinates X."""
    return np.dot(V.reshape(-1), vorpy.symbolic.differential(f, X).reshape(-1)).simplify()

def lie_bracket (A, B, X):
    """
    Compute the Lie bracket of vector fields A and B with respect to coordinates X.

    A formula for [A,B], i.e. the Lie bracket of vector fields A and B, is

        J__B_reshaped*A - J__A_reshaped*B

    where J_X is the Jacobian matrix of the vector field X.

    See https://en.wikipedia.org/wiki/Lie_bracket_of_vector_fields#In_coordinates
    See https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
    """

    if A.shape != B.shape or A.shape != X.shape:
        raise TypeError(f'The shapes of A, B, and X must all be equal, but instead got A.shape = {A.shape}, B.shape = {B.shape}, X.shape = {X.shape}')

    # Flatten the shapes out for computing and contracting the Jacobian matrices.
    A_reshaped = A.reshape(-1)
    B_reshaped = B.reshape(-1)
    X_reshaped = X.reshape(-1)
    # Compute the Jacobian matrices of (the reshaped versions of) A and B
    J__A_reshaped = vorpy.symbolic.differential(A_reshaped, X_reshaped)
    J__B_reshaped = vorpy.symbolic.differential(B_reshaped, X_reshaped)
    # Reshape the result into the original shape of the vector fields and coordinates.
    return (np.dot(J__B_reshaped,A_reshaped) - np.dot(J__A_reshaped,B_reshaped)).reshape(X.shape)

