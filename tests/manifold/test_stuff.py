import numpy as np
import sympy as sp
import vorpy.manifold
import vorpy.symbolic

def lie_bracket__test ():
    n = 3
    X = vorpy.symbolic.tensor('X', (n,))
    X_reshaped = X.reshape(-1)

    A = np.array([
        sp.Function(f'A_{i}')(*list(X_reshaped))
        for i in range(n)
    ])
    B = np.array([
        sp.Function(f'B_{i}')(*list(X_reshaped))
        for i in range(n)
    ])
    #print(f'A = {A}')
    #print(f'B = {B}')
    lb__A_B = vorpy.manifold.lie_bracket(A, B, X)
    #print(f'lb__A_B = {lb__A_B}')

    f = sp.Function('f')(*list(X_reshaped))
    #print(f'f = {f}')

    # Compute the Lie bracket the smart way (just as a function of the vector fields' coordinate expressions),
    # applied to a generic function of the coordinates.
    computed_value = vorpy.manifold.apply_vector_field_to_function(lb__A_B, f, X)
    # Compute the Lie bracket the definitional way (as the commutator of vector fields acting as derivations
    # on functions), applied to a generic function of the coordinates.
    expected_value = vorpy.manifold.apply_vector_field_to_function(A, vorpy.manifold.apply_vector_field_to_function(B, f, X), X) - vorpy.manifold.apply_vector_field_to_function(B, vorpy.manifold.apply_vector_field_to_function(A, f, X), X)

    error = (computed_value - expected_value).simplify()
    #print(f'error in lie brackets (expected value is 0) = {error}')
    if error != 0:
        raise ValueError(f'Error in computed vs expected Lie bracket value was not zero, but instead was {error}')
    print(f'lie_bracket__test passed')

