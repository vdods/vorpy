import numpy as np
import sympy as sp
import vorpy.manifold
import vorpy.symbolic
import vorpy.symplectic

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

def P_x__test ():
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

def H__conservation_test ():
    """
    This test verifies that H is conserved along the flow of H (just a sanity check, this fact
    is easily provable in general).
    """

    qp = phase_space_coordinates()
    #X_H = vorpy.symplectic.symplectic_gradient_of(H(qp), qp)
    H_qp = H__(*qp.reshape(-1).tolist())
    X_H = vorpy.symplectic.symplectic_gradient_of(H_qp, qp)

    # Sanity check
    X_H__H = vorpy.manifold.apply_vector_field_to_function(X_H, H_qp, qp)
    if X_H__H != 0:
        raise ValueError(f'Expected X_H(H) == 0 but instead got {X_H__H}')
    print('H__conservation_test passed')

def p_theta (qp):
    """p_theta is the angular momentum for the system and is conserved along solutions."""

    x,y,z       = qp[0,:]
    p_x,p_y,p_z = qp[1,:]

    return x*p_y - y*p_x

def p_theta__conservation_test ():
    """
    This test verifies that p_theta is conserved along the flow of H.
    """

    qp = phase_space_coordinates()
    X_H = vorpy.symplectic.symplectic_gradient_of(H(qp), qp)

    # Sanity check
    X_H__p_theta = vorpy.manifold.apply_vector_field_to_function(X_H, p_theta(qp), qp)
    if X_H__p_theta != 0:
        raise ValueError(f'Expected X_H(p_theta) == 0 but instead got {X_H__p_theta}')
    print('p_theta__conservation_test passed')

def J (X):
    """J can be thought of as "dilational momentum" for the system, and is conserved along solutions when H = 0."""

    x,y,z       = X[0,:]
    p_x,p_y,p_z = X[1,:]

    return x*p_x + y*p_y + 2*z*p_z

def J__restricted_conservation_test ():
    """This test verifies that J is conserved along the flow of H if restricted to the H = 0 submanifold."""

    qp = phase_space_coordinates()
    H_qp = H(qp)
    X_H = vorpy.symplectic.symplectic_gradient_of(H_qp, qp)

    J_qp = J(qp)
    X_H__J = vorpy.manifold.apply_vector_field_to_function(X_H, J_qp, qp)

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

    print('J__restricted_conservation_test passed')

def J__test ():
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
    X_H__J          = vorpy.manifold.apply_vector_field_to_function(X_H, J_qp, qp)
    #print(f'J__test; X_H__J = {X_H__J}')
    #print(f'J__test; 2*H = {sp.expand(2*H_qp)}')
    actual_value    = X_H__J - sp.expand(2*H_qp)
    #print(f'J__test; X_H__J - 2*H = {actual_value}')

    # Annoyingly, this doesn't simplify to 0 automatically, so some manual manipulation has to be done.

    # Manipulate the expression to ensure the P_x and P_y terms cancel
    actual_value    = sp.collect(actual_value, [P_x_, P_y_])
    #print(f'J__test; after collect P_x, P_y: X_H__J - 2*H = {actual_value}')
    actual_value    = sp.Subs(actual_value, [P_x_, P_y_], [P_x_._expanded(), P_y_._expanded()]).doit()
    #print(f'J__test; after subs P_x, P_y: X_H__J - 2*H = {actual_value}')

    # Manipulate the expression to ensure the mu terms cancel
    actual_value    = sp.factor_terms(actual_value, clear=True, fraction=True)
    #print(f'J__test; after factor_terms: X_H__J - 2*H = {actual_value}')
    actual_value    = sp.collect(actual_value, [r_squared_])
    #print(f'J__test; after collect r_squared_: X_H__J - 2*H = {actual_value}')
    actual_value    = sp.Subs(actual_value, [r_squared_._expanded()], [r_squared_]).doit()
    #print(f'J__test; after subs r_squared: X_H__J - 2*H = {actual_value}')
    actual_value    = sp.Subs(actual_value, [mu_._expanded()], [mu_]).doit()
    #print(f'J__test; after subs mu: X_H__J - 2*H = {actual_value}')

    if actual_value != 0:
        raise ValueError(f'Expected X_H__J - 2*H == 0, but actual value was {actual_value}')

    print('J__test passed')

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

def V__test ():
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
    print('V__test passed')

def lie_bracket_of__X_H__V__test ():
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

    print('lie_bracket_of__X_H__V__test passed')
