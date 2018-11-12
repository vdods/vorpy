import numpy as np
import scipy.linalg
import sympy as sp
import typing
import vorpy.manifold
import vorpy.symbolic
import vorpy.symplectic

def run_and_expect_exception (procedure:typing.Callable[[],None], exception_type:typing.Any) -> None:
    try:
        procedure()
        assert False, f'expected exception of type {exception_type} but did not catch one'
    except BaseException as e:
        assert isinstance(e, exception_type), f'expected exception of type {exception_type}, but caught one of type {type(e)}'

def run_and_expect_no_exception (procedure:typing.Callable[[],None]) -> None:
    try:
        procedure()
    except BaseException as e:
        assert False, f'expected no exception but caught exception {e}'

def test_validate_darboux_coordinates_quantity_or_raise ():
    run_and_expect_exception(lambda:vorpy.symplectic.validate_darboux_coordinates_quantity_or_raise(None), TypeError)
    run_and_expect_exception(lambda:vorpy.symplectic.validate_darboux_coordinates_quantity_or_raise('blah'), TypeError)
    run_and_expect_exception(lambda:vorpy.symplectic.validate_darboux_coordinates_quantity_or_raise(1), TypeError)
    run_and_expect_no_exception(lambda:vorpy.symplectic.validate_darboux_coordinates_quantity_or_raise([4,5]))
    # Doesn't matter that they're not numbers or symbols.
    run_and_expect_no_exception(lambda:vorpy.symplectic.validate_darboux_coordinates_quantity_or_raise(['hippo','ostrich']))
    run_and_expect_no_exception(lambda:vorpy.symplectic.validate_darboux_coordinates_quantity_or_raise([[True],[False]]))

    qp_ = vorpy.symplectic.cotangent_bundle_darboux_coordinates(()) # Empty shape means configuration space is a scalar.
    run_and_expect_no_exception(lambda:vorpy.symplectic.validate_darboux_coordinates_quantity_or_raise(qp_, quantity_name='qp_'))
    for i in range(1,4):
        qp_i = vorpy.symplectic.cotangent_bundle_darboux_coordinates((i,))
        run_and_expect_no_exception(lambda:vorpy.symplectic.validate_darboux_coordinates_quantity_or_raise(qp_i, quantity_name='qp_i'))
        for j in range(1,4):
            qp_ij = vorpy.symplectic.cotangent_bundle_darboux_coordinates((i,j))
            run_and_expect_no_exception(lambda:vorpy.symplectic.validate_darboux_coordinates_quantity_or_raise(qp_ij, quantity_name='qp_ij'))
            for k in range(1,4):
                qp_ijk = vorpy.symplectic.cotangent_bundle_darboux_coordinates((i,j,k))
                run_and_expect_no_exception(lambda:vorpy.symplectic.validate_darboux_coordinates_quantity_or_raise(qp_ijk, quantity_name='qp_ijk'))
    print('test_is_darboux_coordinates_quantity passed')

# TODO: test tautological one form (pullback-canceling property?)

# TODO: test canonical symplectic form (equal to d(tautological_one_form))
# also test that pullback by one-form gives d of that one form (due to commutation of pullback and d)

def test_canonical_symplectic_form_abstract_and_inverse ():
    omega       = vorpy.symplectic.canonical_symplectic_form_abstract(dtype=int)
    omega_inv   = vorpy.symplectic.canonical_symplectic_form_abstract_inverse(dtype=int)
    assert omega.shape == (2,2)
    assert omega_inv.shape == (2,2)
    assert np.all(np.dot(omega, omega_inv) == np.eye(2))
    assert np.all(np.dot(omega_inv, omega) == np.eye(2))
    assert np.all(omega.T == -omega)
    assert np.all(omega.T == omega_inv)
    assert np.all(omega_inv.T == -omega_inv)
    assert np.all(omega_inv.T == omega)
    print('test_canonical_symplectic_form_abstract_and_inverse passed')

def test_canonical_symplectic_form_and_inverse ():
    shape_v = [(2,), (2,1), (2,2), (2,3), (2,1,1), (2,1,2), (2,2,2), (2,3,4), (2,1,1,1), (2,3,4,5)]
    for shape in shape_v:
        dimension_of_shape = vorpy.tensor.dimension_of_shape(shape)
        omega       = vorpy.symplectic.canonical_symplectic_form(shape, dtype=int)
        omega_inv   = vorpy.symplectic.canonical_symplectic_form_inverse(shape, dtype=int)
        assert omega.shape == shape+shape
        assert omega_inv.shape == shape+shape

        input_indices = slice(0, len(shape))
        output_indices = slice(len(shape), 2*len(shape))

        omega_flat = omega.reshape(dimension_of_shape,dimension_of_shape)
        omega_inv_flat = omega_inv.reshape(dimension_of_shape,dimension_of_shape)

        assert np.all(np.dot(omega_flat, omega_inv_flat).reshape(omega.shape) == vorpy.tensor.identity_tensor(shape, dtype=int))
        assert np.all(np.dot(omega_inv_flat, omega_flat).reshape(omega.shape) == vorpy.tensor.identity_tensor(shape, dtype=int))
        assert np.all(omega_flat.T == -omega_flat)
        assert np.all(omega_flat.T == omega_inv_flat)
        assert np.all(omega_inv_flat.T == -omega_inv_flat)
        assert np.all(omega_inv_flat.T == omega_flat)
    print('test_canonical_symplectic_form_and_inverse passed')

def random_antisymmetric_operator_tensor (max_radius:float, operand_space_shape:typing.Tuple[int,...]) -> np.ndarray:
    """
    Uniform distribution on the ball of given radius in the space of antisymmetric matrices, under matrix norm.

    Can only generate dtype=float for now.
    """
    operand_space_dim = vorpy.tensor.dimension_of_shape(operand_space_shape)

    # Antisymmetric in dimension 0 or 1 is identically zero.
    if operand_space_dim <= 1:
        return np.zeros(operand_space_shape+operand_space_shape, dtype=float)

    A_flat_shape = (operand_space_dim,operand_space_dim)
    A_shape = operand_space_shape+operand_space_shape
    while True:
        A_flat = np.random.randn(*A_flat_shape)
        # Project it into antisymmetric matrix space
        A_flat = A_flat - A_flat.T
        norm_A_flat = np.linalg.norm(A_flat)
        # Only accept if the norm is big enough to divide by
        if norm_A_flat > 1.0e-6:
            break

    radius = np.random.uniform(0.0, max_radius)
    A_flat *= radius / norm_A_flat
    return A_flat.reshape(A_shape)

def test_random_antisymmetric_operator_tensor ():
    max_radius = 5.0
    # TODO: Figure out if (0,) is a valid case to test.
    shape_v = [(), (1,), (2,), (2,1), (2,2), (2,3), (2,1,1), (2,3,2), (2,3,4), (2,1,1,1)]
    for shape in shape_v:
        for test_index in range(100):
            A = random_antisymmetric_operator_tensor(max_radius, shape)
            assert np.all(np.isfinite(A))
            assert np.linalg.norm(A) <= max_radius
            A_linop = vorpy.tensor.as_linear_operator(A)
            error = np.max(np.abs(A_linop + A_linop.T))
            assert error < 1.0e-15, f'error = {error}'
    print('test_random_antisymmetric_operator_tensor passed')

def random_rotation_operator_tensor (operand_space_shape:typing.Tuple[int,...]) -> np.ndarray:
    """NOTE: Not a uniform distribution."""
    if vorpy.tensor.dimension_of_shape(operand_space_shape) == 0:
        raise Exception(f'invalid dimension for vector space having rotation')

    A = random_antisymmetric_operator_tensor(np.pi, operand_space_shape)
    return scipy.linalg.expm(vorpy.tensor.as_linear_operator(A)).reshape(A.shape)

def test_random_rotation_operator_tensor ():
    shape_v = [(), (1,), (2,), (2,1), (2,2), (2,3), (2,1,1), (2,3,2), (2,3,4), (2,1,1,1)]
    for shape in shape_v:
        dim = vorpy.tensor.dimension_of_shape(shape)
        for test_index in range(100):
            R = random_rotation_operator_tensor(shape)
            assert np.all(np.isfinite(R))
            assert np.max(np.abs(R)) <= 1.0
            R_linop = vorpy.tensor.as_linear_operator(R)

            error = np.max(np.abs(np.dot(R_linop.T, R_linop) - np.eye(dim, dtype=float)))
            assert error < 1.0e-14, f'error = {error}'

            error = np.max(np.abs(np.dot(R_linop, R_linop.T) - np.eye(dim, dtype=float)))
            assert error < 1.0e-14, f'error = {error}'
    print('test_random_rotation_operator_tensor passed')

def random_invertible_operator_tensor (operand_space_shape:typing.Tuple[int,...]) -> np.ndarray:
    operand_space_dim = vorpy.tensor.dimension_of_shape(operand_space_shape)
    A_shape = operand_space_shape+operand_space_shape
    while True:
        A = np.reshape(np.random.randn(*A_shape), A_shape) # if A_shape is (), then randn returns a scalar.
        A_linop = vorpy.tensor.as_linear_operator(A)
        det_A_linop = np.linalg.det(A_linop)
        if det_A_linop > 1.0e-6:
            break
    return A, det_A_linop

def test_random_invertible_operator_tensor ():
    shape_v = [(), (1,), (2,), (2,1), (2,2), (2,3), (2,1,1), (2,3,2), (2,3,4), (2,1,1,1)]
    for shape in shape_v:
        for test_index in range(100):
            A, returned_det_A_linop = random_invertible_operator_tensor(shape)
            assert np.all(np.isfinite(A))
            A_linop = vorpy.tensor.as_linear_operator(A)
            computed_det_A_linop = np.linalg.det(A_linop)
            det_A_error = np.abs(computed_det_A_linop - returned_det_A_linop)
            assert det_A_error < 1.0e-10, f'det_A_error = {det_A_error}'
            assert returned_det_A_linop > 0.0, f'returned_det_A_linop = {returned_det_A_linop}'
    print('test_random_invertible_operator_tensor passed')

def random_symplectic_lie_algebra_operator_tensor (operand_space_shape:typing.Tuple[int,...]) -> np.ndarray:
    """
    An element of the Lie algebra of the symplectic group has form

        [ P  Q   ]
        [ R -P^T ]

    where R and Q are symmetric.  It is assumed that the symplectic form is in Darboux coordinates, having the form

        [ 0 -I ]
        [ I  0 ]
    """
    vorpy.symplectic.validate_darboux_coordinates_shape_or_raise(operand_space_shape)
    n = vorpy.tensor.dimension_of_shape(operand_space_shape[1:])
    A_darboux = np.random.randn(2,n,2,n)
    # Project into the Lie algebra
    A_darboux[1,:,1,:] = -A_darboux[0,:,0,:].T
    A_darboux[0,:,1,:] += A_darboux[0,:,1,:].T
    A_darboux[0,:,1,:] *= 0.5
    A_darboux[1,:,0,:] += A_darboux[1,:,0,:].T
    A_darboux[1,:,0,:] *= 0.5
    return A_darboux.reshape(operand_space_shape+operand_space_shape)

def test_random_symplectic_lie_algebra_operator_tensor ():
    shape_v = [(2,), (2,1), (2,2), (2,3), (2,1,1), (2,3,2), (2,3,4), (2,1,1,1)]
    for shape in shape_v:
        for test_index in range(100):
            A = random_symplectic_lie_algebra_operator_tensor(shape)
            A_linop = vorpy.tensor.as_linear_operator(A)
            omega = vorpy.symplectic.canonical_symplectic_form(shape, dtype=float)
            omega_linop = vorpy.tensor.as_linear_operator(omega)
            condition = np.dot(omega_linop, A_linop) + np.dot(A_linop.T, omega_linop)
            condition_error = np.max(np.abs(condition))
            assert condition_error < 1.0e-16, f'condition_error = {condition_error}'

    print('test_random_symplectic_lie_algebra_operator_tensor passed')

def random_symplectomorphism_tensor (operand_space_shape:typing.Tuple[int,...]) -> np.ndarray:
    # The Lie group exponential for Sp(2*n, R) is not surjective, but composing two exponentials does reach all
    # elements of Sp(2*n, R).  See https://en.wikipedia.org/wiki/Symplectic_group#Sp(2n,_R)
    X = random_symplectic_lie_algebra_operator_tensor(operand_space_shape)
    Y = random_symplectic_lie_algebra_operator_tensor(operand_space_shape)
    X_linop = vorpy.tensor.as_linear_operator(X)
    Y_linop = vorpy.tensor.as_linear_operator(Y)
    S_linop = np.dot(scipy.linalg.expm(X_linop), scipy.linalg.expm(Y_linop))
    return S_linop.reshape(operand_space_shape+operand_space_shape)

def test_random_symplectomorphism_tensor ():
    shape_v = [(2,), (2,1), (2,2), (2,3), (2,1,1), (2,3,2), (2,3,4), (2,1,1,1)]
    for shape in shape_v:
        for test_index in range(100):
            S = random_symplectomorphism_tensor(shape)
            S_linop = vorpy.tensor.as_linear_operator(S)
            omega = vorpy.symplectic.canonical_symplectic_form(shape, dtype=float)
            omega_linop = vorpy.tensor.as_linear_operator(omega)
            condition = np.einsum('ij,ik,jl', omega_linop, S_linop, S_linop) - omega_linop
            condition_error = np.max(np.abs(condition))
            assert condition_error < 1.0e-7, f'condition_error = {condition_error}'

    print('test_random_symplectomorphism_tensor passed')

def test_symplectomorphicity_condition ():
    np.random.seed(42)

    shape_v = [(2,), (2,1), (2,3), (2,2), (2,1,1), (2,3,2), (2,3,4), (2,1,1,1)]

    # Identity tensor, negative identity tensor, symplectic form, and symplectic form inverse should all
    # be symplectomorphisms.
    for shape in shape_v:
        I = vorpy.tensor.identity_tensor(shape, dtype=int)
        assert np.all(vorpy.symplectic.symplectomorphicity_condition(I, dtype=int) == 0)

        I[...] = -I
        assert np.all(vorpy.symplectic.symplectomorphicity_condition(I, dtype=int) == 0)

        omega = vorpy.symplectic.canonical_symplectic_form(shape, dtype=int)
        assert np.all(vorpy.symplectic.symplectomorphicity_condition(omega, dtype=int) == 0)

        omega_inv = vorpy.symplectic.canonical_symplectic_form_inverse(shape, dtype=int)
        assert np.all(vorpy.symplectic.symplectomorphicity_condition(omega_inv, dtype=int) == 0)

    def rotation_matrix (angle:float) -> np.ndarray:
        return np.array([
            [ np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)],
        ])

    # Rotations in 2d phase space should be symplectomorphisms.
    for angle in np.linspace(0.0, np.pi, 100):
        R = rotation_matrix(angle)
        assert np.max(np.abs(vorpy.symplectic.symplectomorphicity_condition(R, dtype=float))) < 1.0e-15

    # Area-preserving transformations in 2d phase space should be symplectomorphisms.
    for scalar in np.logspace(-3, 3, 10):
        D = np.array([[scalar, 0.0], [0.0, 1.0/scalar]])
        for angle_U in np.linspace(0.0, np.pi, 10):
            U = rotation_matrix(angle_U)
            for angle_V in np.linspace(0.0, np.pi, 10):
                V = rotation_matrix(angle_V)
                A = np.einsum('ij,jk,kl', U, D, V)
                assert np.max(np.abs(vorpy.symplectic.symplectomorphicity_condition(A, dtype=float))) < 1.0e-10

    # Rotations in configuration space (with action induced on phase space) should be symplectomorphisms.
    for test_index in range(100):
        for shape in shape_v:
            dim = vorpy.tensor.dimension_of_shape(shape)
            assert dim % 2 == 0

            config_shape = shape[1:]
            config_R = random_rotation_operator_tensor(config_shape)
            config_R_linop = vorpy.tensor.as_linear_operator(config_R)
            assert np.max(np.abs(np.dot(config_R_linop.T, config_R_linop) - np.eye(dim//2, dtype=float))) < 1.0e-10
            phase_R = np.einsum('ik,jl', np.eye(2, dtype=float), config_R_linop).reshape(shape+shape)
            phase_R_linop = vorpy.tensor.as_linear_operator(phase_R)
            phase_R_error = np.max(np.abs(np.dot(phase_R_linop.T, phase_R_linop) - np.eye(dim, dtype=float)))
            assert phase_R_error < 1.0e-10, f'phase_R_error = {phase_R_error}'

            sympl_error = np.max(np.abs(vorpy.symplectic.symplectomorphicity_condition(phase_R, dtype=float)))
            assert sympl_error < 1.0e-10, f'test_index = {test_index}, sympl_error = {sympl_error}; shape = {shape}, phase_R:\n{phase_R.tolist()}'

    # Symplectomorphisms should be symplectomorphisms.
    #sympl_error_v = []
    for test_index in range(20):
        #print('.', end='', flush=True)
        for shape in shape_v:
            dim = vorpy.tensor.dimension_of_shape(shape)
            assert dim % 2 == 0

            S = random_symplectomorphism_tensor(shape)
            sympl_error = np.max(np.abs(vorpy.symplectic.symplectomorphicity_condition(S, dtype=float)))
            #sympl_error_v.append(sympl_error)
            assert sympl_error < 3.1e-9, f'test_index = {test_index}, sympl_error = {sympl_error}; shape = {shape}, phase_R:\n{phase_R.tolist()}'
    #print()
    #print(f'max sympl error = {np.max(sympl_error_v)}')

    print('test_symplectomorphicity_condition passed')

# TODO: other tests

def phase_space_coordinates ():
    return np.array((
        (  sp.var('x'),   sp.var('y'),   sp.var('z')),
        (sp.var('p_x'), sp.var('p_y'), sp.var('p_z')),
    ))

def P_x (qp):
    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]

    return p_x - y*p_z/2

def P_y (qp):
    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]

    return p_y + x*p_z/2

class FancyFunction(sp.Function):
    def _sympystr (self, printer):
        """Override printer to suppress function evaluation notation; the arguments are understood."""
        if all(arg.is_symbol for arg in self.args):
            return self._name()
        else:
            return f'{self._name()}({",".join(str(arg) for arg in self.args)})'

    def fdiff (self, argindex):
        return self._value(*self.args).diff(self.args[argindex-1])

    def _expanded (self):
        return self._value(*self.args)

class P_x__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'P_x'

    @classmethod
    def _value (cls, x, y, z, p_x, p_y, p_z):
        """Return the expression that defines this function."""
        return p_x - y*p_z/2

    @classmethod
    def eval (cls, x, y, z, p_x, p_y, p_z):
        """Only evaluate special cases where P_x simplifies."""
        value = cls._value(x,y,z,p_x,p_y,p_z)
        if value.is_number or p_x.is_number or y.is_number or p_z.is_number:
            return value
        # NOTE: This function intentionally does NOT return if there is no simplification.

class P_y__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'P_y'

    @classmethod
    def _value (cls, x, y, z, p_x, p_y, p_z):
        """Return the expression that defines this function."""
        return p_y + x*p_z/2

    @classmethod
    def eval (cls, x, y, z, p_x, p_y, p_z):
        # Only evaluate special cases where P_y simplifies.
        value = cls._value(x,y,z,p_x,p_y,p_z)
        if value.is_number or p_y.is_number or x.is_number or p_z.is_number:
            return value
        # NOTE: This function intentionally does NOT return if there is no simplification.

class r_squared__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'r_squared'

    @classmethod
    def _value (cls, x, y):
        """Return the expression that defines this function."""
        return x**2 + y**2

    @classmethod
    def eval (cls, x, y):
        # Only evaluate special cases where r_squared simplifies.
        value = cls._value(x,y)
        if value.is_number or x.is_number or y.is_number:
            return value
        # NOTE: This function intentionally does NOT return if there is no simplification.

class mu__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'mu'

    @classmethod
    def _value (cls, x, y, z):
        """Return the expression that defines this function."""
        r_squared_ = r_squared__(x,y)
        return r_squared_**2 + 16*z**2

    @classmethod
    def eval (cls, x, y, z):
        # Only evaluate special cases where mu simplifies.
        value = cls._value(x,y,z)
        if value.is_number or r_squared__.eval(x,y) is not None or z.is_number:
            return value
        # NOTE: This function intentionally does NOT return if there is no simplification.

class K__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'K'

    @classmethod
    def _value (cls, x, y, z, p_x, p_y, p_z):
        """Return the expression that defines this function."""
        return (P_x__(x,y,z,p_x,p_y,p_z)**2 + P_y__(x,y,z,p_x,p_y,p_z)**2)/2

    @classmethod
    def eval (cls, x, y, z, p_x, p_y, p_z):
        # Only evaluate special cases where mu simplifies.
        value = cls._value(x,y,z,p_x,p_y,p_z)
        #if value.is_number or P_x__.eval(x,y,z,p_x,p_y,p_z) is not None or P_y__.eval(x,y,z,p_x,p_y,p_z) is not None:
            #return value
        ## NOTE: This function intentionally does NOT return if there is no simplification.
        # Always evaluate
        return value

class U__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'U'

    @classmethod
    def _value (cls, x, y, z, p_x, p_y, p_z):
        """Return the expression that defines this function."""
        return -1 / (8*sp.pi*sp.sqrt(mu__(x,y,z)))

    @classmethod
    def eval (cls, x, y, z, p_x, p_y, p_z):
        # Only evaluate special cases where mu simplifies.
        value = cls._value(x,y,z,p_x,p_y,p_z)
        #if value.is_number or mu__.eval(x,y,z) is not None:
            #return value
        ## NOTE: This function intentionally does NOT return if there is no simplification.
        # Always evaluate
        return value

class H__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'H'

    @classmethod
    def _value (cls, x, y, z, p_x, p_y, p_z):
        """Return the expression that defines this function."""
        return K__(x,y,z,p_x,p_y,p_z) + U__(x,y,z,p_x,p_y,p_z)

    @classmethod
    def eval (cls, x, y, z, p_x, p_y, p_z):
        # Only evaluate special cases where mu simplifies.
        value = cls._value(x,y,z,p_x,p_y,p_z)
        #if value.is_number or K__.eval(x,y,z,p_x,p_y,p_z) is not None or U__.eval(x,y,z,p_x,p_y,p_z) is not None:
            #return value
        ## NOTE: This function intentionally does NOT return if there is no simplification.
        # Always evaluate
        return value

class J__(FancyFunction):
    @classmethod
    def _name (cls):
        return 'J'

    @classmethod
    def _value (cls, x, y, z, p_x, p_y, p_z):
        """Return the expression that defines this function."""
        return x*p_x + y*p_y + 2*z*p_z

    @classmethod
    def eval (cls, x, y, z, p_x, p_y, p_z):
        # Only evaluate special cases where mu simplifies.
        value = cls._value(x,y,z,p_x,p_y,p_z)
        #if value.is_number or any(v.is_number for v in (x,y,z,p_x,p_y,p_z)):
            #return value
        ## NOTE: This function intentionally does NOT return if there is no simplification.
        # Always evaluate
        return value

def test_P_x ():
    # TODO: Deprecate this, it's just to test how subclassing sp.Function works.

    qp = phase_space_coordinates()

    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]

    #P_x_ = P_x__(qp)
    #P_x_ = P_x__(x,y,z,p_x,p_y,p_z)
    P_x_ = P_x__(*qp.reshape(-1).tolist())
    print(f'P_x_ = {P_x_}')
    print(f'P_x__(x,y,z,p_x,p_y,p_z) = {P_x__(x,y,z,p_x,p_y,p_z)}')
    print(f'P_x__(x,0,z,p_x,p_y,p_z) = {P_x__(x,0,z,p_x,p_y,p_z)}')

    print(f'P_x_.diff(x) = {P_x_.diff(x)}')
    print(f'P_x_.diff(y) = {P_x_.diff(y)}')
    print(f'P_x_.diff(z) = {P_x_.diff(z)}')
    print(f'P_x_.diff(p_x) = {P_x_.diff(p_x)}')
    print(f'P_x_.diff(p_y) = {P_x_.diff(p_y)}')
    print(f'P_x_.diff(p_z) = {P_x_.diff(p_z)}')
    print(f'P_x_.diff(qp) = {P_x_.diff(qp)}')

    mu_ = mu__(*qp.reshape(-1).tolist()[:3])
    print(f'mu_ = {mu_}, mu_.func = {mu_.func}')
    print(f'mu__(x,y,0) = {mu__(x,y,0)}')
    print(f'mu__(x,0,z) = {mu__(x,0,z)}')

    K = (P_x__(*qp.reshape(-1).tolist())**2 + P_y__(*qp.reshape(-1).tolist())**2)/2
    print(f'K = {K}')

    U = -1 / (8*sp.pi*sp.sqrt(mu_))
    print(f'U = {U}')

    #H = K + U
    H = H__(*qp.reshape(-1).tolist())
    print(f'H = {H}')
    H_diff = H.diff(qp)
    print(f'H.diff(qp) = {H_diff}, type(H.diff(qp)) = {type(H_diff)}')

    dH = vorpy.symbolic.differential(H, qp)
    print(f'dH = {dH}')
    print(f'symplectic gradient of H = {vorpy.symplectic.symplectic_gradient_of(H, qp)}')

def K (qp):
    return (P_x(qp)**2 + P_y(qp)**2)/2

def r_squared (qp):
    x,y,z       = qp[0,:]

    return x**2 + y**2

def mu (qp):
    x,y,z       = qp[0,:]

    beta        = sp.Integer(16)

    return r_squared(qp)**2 + beta*z**2

def U (qp):
    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]

    alpha       = 1 / (8*sp.pi)

    return -alpha * mu(qp)**sp.Rational(-1,2)

def H (qp):
    """H is the Hamiltonian for the system."""

    return K(qp) + U(qp)

def test_H_conservation ():
    """
    This test verifies that H is conserved along the flow of H (just a sanity check, this fact
    is easily provable in general).
    """

    qp = phase_space_coordinates()
    #X_H = vorpy.symplectic.symplectic_gradient_of(H(qp), qp)
    H_qp = H__(*qp.reshape(-1).tolist())
    X_H = vorpy.symplectic.symplectic_gradient_of(H_qp, qp)

    # Sanity check
    X_H__H = vorpy.manifold.directional_derivative(X_H, H_qp, qp)
    if X_H__H != 0:
        raise ValueError(f'Expected X_H(H) == 0 but instead got {X_H__H}')
    print('test_H_conservation passed')

def p_theta (qp):
    """p_theta is the angular momentum for the system and is conserved along solutions."""

    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]

    return x*p_y - y*p_x

def test_p_theta_conservation ():
    """
    This test verifies that p_theta is conserved along the flow of H.
    """

    qp = phase_space_coordinates()
    X_H = vorpy.symplectic.symplectic_gradient_of(H(qp), qp)

    # Sanity check
    X_H__p_theta = vorpy.manifold.directional_derivative(X_H, p_theta(qp), qp)
    if X_H__p_theta != 0:
        raise ValueError(f'Expected X_H(p_theta) == 0 but instead got {X_H__p_theta}')
    print('test_p_theta_conservation passed')

def J (X):
    """J can be thought of as "dilational momentum" for the system, and is conserved along solutions when H = 0."""

    x,y,z       = X[0,:]
    p_x,p_y,p_z = X[1,:]

    return x*p_x + y*p_y + 2*z*p_z

def test_J_restricted_conservation ():
    """This test verifies that J is conserved along the flow of H if restricted to the H = 0 submanifold."""

    qp = phase_space_coordinates()
    H_qp = H(qp)
    X_H = vorpy.symplectic.symplectic_gradient_of(H_qp, qp)

    J_qp = J(qp)
    X_H__J = vorpy.manifold.directional_derivative(X_H, J_qp, qp)

    p_z = qp[1,2]

    # Solve for p_z in H_qp == 0; there are two sheets to this solution.
    p_z_solution_v = sp.solve(H_qp, p_z)
    assert len(p_z_solution_v) == 2, f'Expected 2 solutions for p_z in H == 0, but instead got {len(p_z_solution_v)}'
    #print('There are {0} solutions for the equation: {1} = 0'.format(len(p_z_solution_v), H_qp))
    #for i,p_z_solution in enumerate(p_z_solution_v):
        #print('    solution {0}: p_z = {1}'.format(i, p_z_solution))

    for solution_index,p_z_solution in enumerate(p_z_solution_v):
        # We have to copy X_H__J or it will only be a view into X_H__J and will modify the original.
        # The [tuple()] access is to obtain the scalar value out of the
        # sympy.tensor.array.dense_ndim_array.ImmutableDenseNDimArray object that doit() returns.
        X_H__J__restricted = sp.Subs(np.copy(X_H__J), p_z, p_z_solution).doit()[tuple()].simplify()
        #print(f'solution_index = {solution_index}, X_H__J__restricted = {X_H__J__restricted}')
        if X_H__J__restricted != 0:
            raise ValueError(f'Expected X_H__J__restricted == 0 for solution_index = {solution_index}, but actual value was {X_H__J__restricted}')

    print('test_J_restricted_conservation passed')

def test_J ():
    """This test verifies that dJ/dt = 2*H."""

    qp              = phase_space_coordinates()
    qp_             = qp.reshape(-1).tolist()

    x,y,z           = qp[0,:]
    p_x,p_y,p_z     = qp[1,:]

    P_x_            = P_x__(*qp_)
    P_y_            = P_y__(*qp_)
    mu_             = mu__(x,y,z)
    r_squared_      = r_squared__(x,y)

    H_qp            = H__(*qp_)
    X_H             = vorpy.symplectic.symplectic_gradient_of(H_qp, qp)

    J_qp            = J__(*qp_)
    # Because X_H gives the vector field defining the time derivative of a solution to the dynamics,
    # it follows that X_H applied to J is equal to dJ/dt (where J(t) is J(qp(t)), where qp(t) is a
    # solution to Hamilton's equations).
    X_H__J          = vorpy.manifold.directional_derivative(X_H, J_qp, qp)
    #print(f'test_J; X_H__J = {X_H__J}')
    #print(f'test_J; 2*H = {sp.expand(2*H_qp)}')
    actual_value    = X_H__J - sp.expand(2*H_qp)
    #print(f'test_J; X_H__J - 2*H = {actual_value}')

    # Annoyingly, this doesn't simplify to 0 automatically, so some manual manipulation has to be done.

    # Manipulate the expression to ensure the P_x and P_y terms cancel
    actual_value    = sp.collect(actual_value, [P_x_, P_y_])
    #print(f'test_J; after collect P_x, P_y: X_H__J - 2*H = {actual_value}')
    actual_value    = sp.Subs(actual_value, [P_x_, P_y_], [P_x_._expanded(), P_y_._expanded()]).doit()
    #print(f'test_J; after subs P_x, P_y: X_H__J - 2*H = {actual_value}')

    # Manipulate the expression to ensure the mu terms cancel
    actual_value    = sp.factor_terms(actual_value, clear=True, fraction=True)
    #print(f'test_J; after factor_terms: X_H__J - 2*H = {actual_value}')
    actual_value    = sp.collect(actual_value, [r_squared_])
    #print(f'test_J; after collect r_squared_: X_H__J - 2*H = {actual_value}')
    actual_value    = sp.Subs(actual_value, [r_squared_._expanded()], [r_squared_]).doit()
    #print(f'test_J; after subs r_squared: X_H__J - 2*H = {actual_value}')
    actual_value    = sp.Subs(actual_value, [mu_._expanded()], [mu_]).doit()
    #print(f'test_J; after subs mu: X_H__J - 2*H = {actual_value}')

    if actual_value != 0:
        raise ValueError(f'Expected X_H__J - 2*H == 0, but actual value was {actual_value}')

    print('test_J passed')

def A (qp):
    """
    A is the standard contact form in R^3, taken as a differential form in T*(R^3), then V is its symplectic dual.

    A = dz + y/2 * dx - x/2 * dy
    """
    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]
    return np.array((
        (          y/2,          -x/2, sp.Integer(1)),
        (sp.Integer(0), sp.Integer(0), sp.Integer(0)),
    ))

def V (qp):
    """
    If A is the standard contact form in R^3, taken as a differential form in T*(R^3), then V is its symplectic dual.

    A = dz + y/2 * dx - x/2 * dy
    V = del_{p_z} + y/2 * del_{p_x} - x/2 * del_{p_y}
    """
    return vorpy.symplectic.symplectic_dual_of_covector_field(A(qp))

def test_V ():
    qp = phase_space_coordinates()

    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]

    expected_value = np.array((
        (sp.Integer(0), sp.Integer(0), sp.Integer(0)),
        (         -y/2,           x/2,            -1),
    ))
    actual_value = V(qp)
    error = actual_value - expected_value
    if not np.all(error == 0):
        raise ValueError(f'Expected V = {expected_value} but it was actually {actual_value}')
    print('test_V passed')

def test_lie_bracket_of__X_H__V ():
    qp = phase_space_coordinates()
    #print(f'H = {H(qp)}')
    #print(f'X_H = {vorpy.symplectic.symplectic_gradient_of(H(qp), qp)}')

    #print(f'A = {A(qp)}')
    #print(f'V = {V(qp)}')

    lb__X_H__V = vorpy.manifold.lie_bracket(vorpy.symplectic.symplectic_gradient_of(H(qp), qp), V(qp), qp)
    #print(f'[X_H,V] = {lb__X_H__V}')

    # NOTE: This has sign opposite from Corey's Weinstein Note PDF file (he has a sign
    # error in computing the symplectic dual of V).
    expected__lb__X_H__V = np.array((
        (sp.Integer(0), sp.Integer(0), sp.Integer(0)),
        (     -P_y(qp),       P_x(qp), sp.Integer(0)),
    ))
    #print(f'expected value = {expected__lb__X_H__V}')
    #print(f'[X_H,V] - expected_value = {lb__X_H__V - expected__lb__X_H__V}')

    if not np.all(lb__X_H__V == expected__lb__X_H__V):
        raise ValueError(f'Expected [X_H,V] = {expected__lb__X_H__V} but it was actually {lb__X_H__V}')

    print('test_lie_bracket_of__X_H__V passed')

if __name__ == '__main__':
    # Just run the tests in this module
    # TODO: Figure out how to do this automatically with nosetest
    test_validate_darboux_coordinates_quantity_or_raise()
    test_canonical_symplectic_form_abstract_and_inverse()
    test_canonical_symplectic_form_and_inverse()
    test_random_antisymmetric_operator_tensor()
    test_random_rotation_operator_tensor()
    test_random_invertible_operator_tensor()
    test_random_symplectic_lie_algebra_operator_tensor()
    test_random_symplectomorphism_tensor()
    test_symplectomorphicity_condition()
    if False:
        test_P_x()
        test_H_conservation()
        test_p_theta_conservation()
        test_J_restricted_conservation()
        test_J()
        test_V()
        test_lie_bracket_of__X_H__V()
