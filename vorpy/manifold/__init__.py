import numpy as np
import sympy as sp
import typing
import vorpy.symbolic

def directional_derivative (
    V:np.ndarray,
    f:typing.Any,
    X:np.ndarray,
    *,
    post_process_o:typing.Optional[typing.Callable[[typing.Any],typing.Any]]=sp.simplify,
) -> np.ndarray:
    """
    This returns the directional derivative of f along V in coordinates X.

    If post_process_o is not None, will call post_process_o on each component of the result.
    The default for post_process_o is sympy.simplify.
    """
    if V.shape != X.shape:
        raise TypeError(f'expected vector field V to have the same shape as coordinates X, but V.shape = {V.shape} and X.shape = {X.shape}')

    V_flat = V.reshape(-1)
    f_flat = np.reshape(f, -1)
    X_flat = X.reshape(-1)
    df_flat = vorpy.symbolic.differential(f_flat, X_flat)
    assert df_flat.shape == f_flat.shape + X_flat.shape
    V_dot_df = np.dot(df_flat, V_flat).reshape(np.shape(f))

    # For some reason, the result of sp.simplify(V_dot_df) is class
    # 'sympy.tensor.array.dense_ndim_array.ImmutableDenseNDimArray'
    # so instead call sp.simplify on each element of the result so
    # that it is a numpy.ndarray.  If the order of the result is
    # zero, then extract the single, scalar element.
    if post_process_o is not None:
        V_dot_df = np.vectorize(post_process_o)(V_dot_df)

    # TODO: test on scalar expressions f, as this [()] expression may not work in that case
    if V_dot_df.shape == ():
        return V_dot_df[()]
    else:
        return V_dot_df

def lie_bracket (A, B, X):
    """
    Compute the Lie bracket of vector fields A and B with respect to coordinates X.

    A formula for [A,B], i.e. the Lie bracket of vector fields A and B, is

        J__B_reshaped*A - J__A_reshaped*B

    where J_V is the Jacobian matrix of the vector field V.

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

