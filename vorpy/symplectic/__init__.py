"""
This module contains various elements of symplectic geometry.

Let "shape" refer to the numpy.ndarray concept of shape, which in particular is a tuple (of any length) having
nonnegative integer elements.  The shape of an ndarray corresponds to a coordinatized expression for a tensor
whose factors have the dimensions given by the components of that shape.

Generally things will be expressed in Darboux coordinates, due to Darboux's theorem.  Darboux coordinates will
be denoted as `qp`.  An important piece of terminology used in this module is the concept of what will be called
the "symplectic axis".  When expressing a quantity in Darboux coordinates (which could be a vector field, covector
field, or even Darboux coordinates themselves), it generally has the shape

    (2,)+configuration_space_shape,

where configuration_space_shape is a shape (a tuple having nonnegative integer elements, possibly empty).  Here,
"shape" refers to the numpy.ndarray concept of shape.  Take note:

    The axis which distinguishes position coordinates (component 0) from momentum coordinates (component 1) within
    a quantity expressed in Darboux coordinates (therefore having dimension 2) will be referred to as the
    "symplectic axis" for that quantity.

Most theory regarding symplectic geometry assumes the coordinates have the shape (2,n) for some n > 0, since
in the general theory there's no reason to distinguish the different position coordinates.  However, in specific
situations, there might be reason for the underlying configuration space to have a shape more specific than
just (n,).

Note that all this talk about configuration/phase space and position/momentum is due to the heavy bias toward
use of the cotangent bundle as the archetypical symplectic manifold.  The Darboux theorem makes this a
reasonable thing to do.

It's also possible for a quantity expressed in Darboux coordinates to have shape (2,), meaning that the position
and momentum coordinates are each scalar values.

References
-   Darboux coordinates: https://en.wikipedia.org/wiki/Darboux%27s_theorem
-   https://en.wikipedia.org/wiki/Tautological_one-form
-   Choice of convention for the canonical symplectic form:
    https://symplecticfieldtheorist.wordpress.com/2015/08/23/signs-or-how-to-annoy-a-symplectic-topologist/

TODO: Maybe move this under vorpy.manifold, since symplectic linear algebra could be thought
to be separate and could go under vorpy.linalg.
"""

import numpy as np
import typing
import vorpy.linalg
import vorpy.symbolic
import vorpy.tensor

def is_darboux_coordinates_shape (shape:typing.Tuple[int,...]) -> bool:
    return len(shape) >= 1 and shape[0] == 2

def validate_darboux_coordinates_shape_or_raise (shape:typing.Tuple[int,...], *, shape_name:str='shape') -> None:
    if not is_darboux_coordinates_shape(shape):
        raise TypeError(f'{shape_name} was expected to be a shape representing Darboux coordinates (i.e. be (2,)+s for some shape s), but {shape_name} was actually {shape}')

def is_darboux_coordinates_quantity (quantity:typing.Any) -> bool:
    return is_darboux_coordinates_shape(np.shape(quantity))

def validate_darboux_coordinates_quantity_or_raise (quantity:typing.Any, *, quantity_name:str='quantity') -> None:
    if not is_darboux_coordinates_quantity(quantity):
        raise TypeError(f'{quantity_name} was expected to be a quantity expressed in Darboux coordinates (i.e. have shape (2,)+s for some shape s), but its shape was actually {np.shape(quantity)}')

def validate_symplectic_axis_of_shape (symplectic_axis:int, shape:typing.Tuple[int,...], *, symplectic_axis_name:str='symplectic_axis', shape_name:str='shape_name') -> None:
    vorpy.tensor.validate_shape_or_raise(shape, shape_name)
    if not (-len(shape) <= symplectic_axis < len(shape)):
        raise TypeError(f'expected {symplectic_axis_name} a valid axis index for shape {shape_name} = {shape} (i.e. {-len(shape)} <= {symplectic_axis_name} < {len(shape)}; negative index used as in Python), but {symplectic_axis_name} was actually {symplectic_axis}')
    if shape[symplectic_axis] != 2:
        raise TypeError(f'expected {symplectic_axis_name} to specify an index distinguishing position coordinates from momentum coordinates (which in particular must have dimension 2), but axis {symplectic_axis} has dimension {shape[symplectic_axis]}')

def cotangent_bundle_darboux_coordinates (configuration_space_shape:typing.Tuple[int,...]) -> np.ndarray:
    """
    Returns symbolic Darboux coordinates for T^{*}Q, where Q is the configuration space, and where coordinates
    on Q have the shape given by configuration_space_shape.  The return value will have shape

        (2,)+configuration_space_shape

    where the symplectic axis is 0.  If qp denotes the returned value, then position coordinates will be
    qp[0,...], having names q_I, while the momentum coordinates will be qp[1,...], having names p_J, where
    I and J are multiindices for shape configuration_space_shape.  In particular, the way a multiindex I
    is rendered is as '_'.join(str(i) for i in I) (i.e. underscore-separated component indices).
    """

    vorpy.tensor.validate_shape_or_raise(configuration_space_shape, 'configuration_space_shape')
    retval = np.ndarray((2,)+configuration_space_shape, dtype=object)
    retval[0,...] = vorpy.symbolic.tensor('q', configuration_space_shape)
    retval[1,...] = vorpy.symbolic.tensor('p', configuration_space_shape)
    return retval

def positions (qp:np.ndarray) -> np.ndarray:
    """
    Returns the `position` coordinates from a Darboux coordinate quantity (this is using the cotangent bundle
    archetype for symplectic manifold).
    """
    validate_darboux_coordinates_quantity_or_raise(qp, quantity_name='qp')
    return qp[0,...]

def momenta (qp:np.ndarray) -> np.ndarray:
    """
    Returns the `momentum` coordinates from a Darboux coordinate quantity (this is using the cotangent bundle
    archetype for symplectic manifold).
    """
    validate_darboux_coordinates_quantity_or_raise(qp, quantity_name='qp')
    return qp[1,...]

def tautological_one_form (qp:np.ndarray) -> np.ndarray:
    """
    Returns the tautological one form on the cotangent bundle T^{*}Q having Darboux coordinates qp,
    where qp has shape (2,...), and qp[0,...] are the position coordinates and qp[1,...] are
    the momentum coordinates.

    In particular, the tautological one form is given by the [pseudocode] expression

        sum(p[i] * dq[i] for i in range(n))

    where n is dim(Q), and q[i] and p[i] refer to the Darboux coordinates as if qp had shape (2,n).

    NOTE: This assumes that the symplectic_axis value is 0.

    TODO: Talk about pullback-canceling property
    """
    validate_darboux_coordinates_quantity_or_raise(qp, quantity_name='qp')
    # Make a copy of qp so the shape and dtype of the return value are correct.
    retval = np.copy(qp)
    # Put the p[i] coordinates in the dq[i] slots.
    retval[0,...] = retval[1,...]
    # Set the dp[i] slots equal to 0 (using a hackish but effective way to get the correct type of zero
    # without specifying dtype).
    retval[1,...] -= retval[1,...]
    return retval

def canonical_symplectic_form_abstract (*, dtype:typing.Any) -> np.ndarray:
    """
    Returns an abstract version of the canonical symplectic form on the cotangent bundle M := T^{*}Q in
    Darboux coordinates.  What is meant by "abstract version" is that the returned value is a 2x2 matrix

        [ 0 -1 ]
        [ 1  0 ]

    which can be contracted with the symplectic axis of an expression in Darboux coordinates.  See below for
    a practical description of how this works.

    NOTE: There is a convention to be chosen when defining the canonical symplectic form, and the convention
    chosen here is the one where Hamilton's equations corresponding to the Hamiltonian function H are

        d(qp)/dt = symplectic_gradient_of(H(qp), qp)

    where the symplectic gradient of a function H is the inverse of the symplectic form contracted with the
    one form dH (where dH is contracted with the `right side` of the inverse of the symplectic form).  For
    reference, Hamilton's equations are ordinarily written as

        dq/dt =  dH/dp
        dp/dt = -dH/dq

    where dH/dq and dH/dp denote the partial derivatives of H with respect to q and p respectively.  In particular,

        canonical_symplectic_form_abstract = d(tautological_one_form).

    As a side note, applying this form to a vector written as q+i*p can be thought of as multiplying by the
    imaginary number i.

    The canonical symplectic form is a section of

        T^{*}M \wedge T^{*}M

    or can be thought of (as it is used here) as an alternating section of

        T^{*}M \otimes T^{*}M

    and therefore "naturally converts" a vector field on M (i.e. a section of TM) into a covector field on M
    (i.e. a section of T^{*}M).

    A practical example of use of this "abstract version" of the canonical symplectic form is the following.
    Let w denote the abstract version of the canonical symplectic form.  Let H be a symbolic expression in
    Darboux coordinates qp having shape (2,n) so that the symplectic_axis is 0.  Then dH, the differential
    of H has shape (2,n) and the symplectic gradient of H is given by the [pseudocode] expression

        sum(w[i,j] * dH[j,k] for j in range(2))

    which is an index expression in indices i,k having shape (2,n), and represents a vector field on M, meaning
    it is a section of TM.

    Reference(s)
    -   https://symplecticfieldtheorist.wordpress.com/2015/08/23/signs-or-how-to-annoy-a-symplectic-topologist/
    """
    retval = np.zeros((2,2), dtype=dtype)
    retval[0,1] = dtype(-1)
    retval[1,0] = dtype( 1)
    return retval

def canonical_symplectic_form_abstract_inverse (*, dtype:typing.Any) -> np.ndarray:
    """
    Returns the inverse of canonical_symplectic_form_abstract(dtype=dtype).  See documentation for that function for more.
    In particular, the inverse of the [abstract version of the] canonical symplectic form is

        [  0 1 ]
        [ -1 0 ]

    The inverse of the canonical symplectic form is a section of

        TM \wedge TM

    or can be thought of (as it is used here) as an alternating section of

        TM \otimes TM

    and therefore "naturally converts" a covector field on M (i.e. a section of T^{*}M) into a vector field on M
    (i.e. a section of TM).

    This form is what's used in the definition of the symplectic gradient of a function.
    """
    retval = np.zeros((2,2), dtype=dtype)
    retval[0,1] = dtype( 1)
    retval[1,0] = dtype(-1)
    return retval

def canonical_symplectic_form (darboux_coordinates_shape:typing.Tuple[int,...], *, dtype:typing.Any) -> np.ndarray:
    """
    Returns the canonical symplectic form on the cotangent bundle M := T^{*}Q in the given Darboux coordinates
    (where the coordinates only need to be specified by their shape).  In particular, if there are 2*n coordinates,
    having shape (2,)+s for some shape s (s is the shape of the configuration space), and if the coordinates were
    reshaped to (2*n,), then the canonical symplectic form would be

        [ 0 -I ]
        [ I  0 ]

    However, this description involving reshaping is only so that the quantity above can be a 2-tensor (mairix)
    instead of a 2*len(s) tensor, which is what it actually will be.

    In particular, if d denotes darboux_coordinates_shape, then the shape of the returned value will be d+d (i.e.
    the concatenation of d with itself).  This represents a linear operator acting on a tensor space having shape
    d.

    NOTE: There is a convention to be chosen when defining the canonical symplectic form, and the convention
    chosen here is the one where Hamilton's equations corresponding to the Hamiltonian function H are

        d(qp)/dt = symplectic_gradient_of(H(qp), qp)

    where the symplectic gradient of a function H is the inverse of the symplectic form contracted with the
    one form dH (where dH is contracted with the `right side` of the inverse of the symplectic form).  For
    reference, Hamilton's equations are ordinarily written as

        dq/dt =  dH/dp
        dp/dt = -dH/dq

    where dH/dq and dH/dp denote the partial derivatives of H with respect to q and p respectively.  In particular,

        canonical_symplectic_form = d(tautological_one_form).

    As a side note, applying this form to a vector written as q+i*p can be thought of as multiplying by the
    imaginary number i.

    The canonical symplectic form is a section of

        T^{*}M \wedge T^{*}M

    or can be thought of (as it is used here) as an alternating section of

        T^{*}M \otimes T^{*}M

    and therefore "naturally converts" a vector field on M (i.e. a section of TM) into a covector field on M
    (i.e. a section of T^{*}M).
    """
    validate_darboux_coordinates_shape_or_raise(darboux_coordinates_shape)
    assert vorpy.tensor.dimension_of_shape(darboux_coordinates_shape) % 2 == 0
    configuration_space_dimension = vorpy.tensor.dimension_of_shape(darboux_coordinates_shape) // 2
    omega = vorpy.tensor.contract(
        'ik,jl',
        canonical_symplectic_form_abstract(dtype=dtype),
        np.eye(configuration_space_dimension, dtype=dtype),
        dtype=dtype,
    )
    assert omega.shape == (2,configuration_space_dimension,2,configuration_space_dimension)
    return omega.reshape(darboux_coordinates_shape+darboux_coordinates_shape)

def canonical_symplectic_form_inverse (darboux_coordinates_shape:typing.Tuple[int,...], *, dtype:typing.Any) -> np.ndarray:
    """
    Returns the inverse of canonical_symplectic_form(dtype=dtype).  See documentation for that function for more.
    In particular, the inverse of the canonical symplectic form is

        [  0 I ]
        [ -I 0 ]

    The inverse of the canonical symplectic form is a section of

        TM \wedge TM

    or can be thought of (as it is used here) as an alternating section of

        TM \otimes TM

    and therefore "naturally converts" a covector field on M (i.e. a section of T^{*}M) into a vector field on M
    (i.e. a section of TM).

    This form is what's used in the definition of the symplectic gradient of a function.
    """
    validate_darboux_coordinates_shape_or_raise(darboux_coordinates_shape)
    assert vorpy.tensor.dimension_of_shape(darboux_coordinates_shape) % 2 == 0
    configuration_space_dimension = vorpy.tensor.dimension_of_shape(darboux_coordinates_shape) // 2
    omega_inv = vorpy.tensor.contract(
        'ik,jl',
        canonical_symplectic_form_abstract_inverse(dtype=dtype),
        np.eye(configuration_space_dimension, dtype=dtype),
        dtype=dtype,
    )
    assert omega_inv.shape == (2,configuration_space_dimension,2,configuration_space_dimension)
    return omega_inv.reshape(darboux_coordinates_shape+darboux_coordinates_shape)

def symplectomorphicity_condition (A:np.ndarray, *, dtype:typing.Any, return_as_scalar_if_possible:bool=False) -> np.ndarray:
    """
    This returns a tensor quantifying the failure of the linear operator A to be a symplectomorphism.
    In particular, A is a symplectomorphism if and only if the returned value is identically 0.

    A is expected to be a tensor having shape (2,)+s+(2,)+s for some shape s (possibly empty), meaning that A
    linearly operates on a tensor space of shape (2,)+s.  It is assumed that the operand space coordinates
    are Darboux coordinates.

    Returns a numpy.ndarray having shape equal to A quantifying the failure of this condition -- A is a
    symplectomorphism if and only if the failure value is identically zero.  The returned quantity is
    antisymmetric (with respect to the symplectic axes).

    If vorpy.tensor.dimension_of_shape((2,)+s) == 2, then because the return value is an antisymmetric
    tensor having 4 components, there is only 1 degree of freedom, and the value can be collapsed into
    a scalar, which will happen only if return_as_scalar_if_possible is True.

    TODO: Really this function is not specific to symplectomorphisms, it works in the category of symplectic
    vector spaces as well.
    """

    operand_shape = vorpy.tensor.operand_shape_of(A)
    if len(operand_shape) == 0 or operand_shape[0] != 2:
        raise Exception(f'expected operand shape to be (2,...), but it was actually {operand_shape}')

    # s is the s from the docstring.
    s = operand_shape[1:]
    n = vorpy.tensor.dimension_of_shape(s)

    # Square matrix operating on vectors having shape (2,n)
    A_darboux = A.reshape(2,n,2,n)

    omega_abstract = canonical_symplectic_form_abstract(dtype=dtype)
    omega = canonical_symplectic_form(operand_shape, dtype=dtype)
    omega_darboux = omega.reshape(2,n,2,n)
    # The condition is that the pullback of omega by A is omega.
    condition_darboux = vorpy.tensor.contract('ij,ikpq,jkxy', omega_abstract, A_darboux, A_darboux, dtype=dtype) - omega_darboux

    if False:
        # Equivalent formulation as a sanity check
        A_linop = vorpy.tensor.as_linear_operator(A)
        omega_linop = vorpy.tensor.as_linear_operator(omega)
        condition_linop = vorpy.tensor.contract('ij,ik,jl', omega_linop, A_linop, A_linop, dtype=dtype) - omega_linop

        condition_darboux_linop = vorpy.tensor.as_linear_operator(condition_darboux)
        comp_error = np.max(np.abs(condition_darboux_linop - condition_linop))
        assert comp_error < 1.0e-10, f'comp_error = {comp_error}'

    # Reshape the condition to be in the expected shape.
    condition = condition_darboux.reshape(operand_shape+operand_shape)

    # Because the diagonal of the condition is zero (the condition is an antisymmetric quantity), in the n == 1 case, the
    # condition is an antisymmetric 2x2 matrix, and therefore only has 1 degree of freedom.  Thus it is possible to return
    # it as a scalar value.  If return_as_scalar_if_possible is True, then return this 1 independent quantity.
    if n == 1 and return_as_scalar_if_possible:
        assert condition.shape == (2,2)
        return condition[1,0] # Just return one of the off diagonals; doesn't matter which, since condition[1,0] == -condition[0,1]
    else:
        return condition

# TODO: Perhaps make an `exterior calculus` module which defines the d and wedge operators

def symplectic_dual_of_vector_field_with_extra_factors (V, *, symplectic_axis:int, dtype:typing.Any=int) -> np.ndarray:
    """
    This is a version of symplectic_dual_of_vector_field where the field can have extra factors whose types
    are left unchanged by the operation.  The symplectic_axis must be specified because there are fewer assumptions
    about the shape of V in this function.

    For example, if V has shape (4,5,2,6,3), where symplectic_axis == 2, and (6,3) refers to the shape of the
    configuration space, then the "extra factors" have shape (4,5), and the expression for this function would be

        symplectic_dual_of_vector_field_with_extra_factors(V, symplectic_axis=2)

    In terms of tensor field types, if V is a section of

        F \otimes TM

    where M is the symplectic manifold in question, then the symplectic dual of V is a section of

        F \otimes T^{*}M

    in other words, replacing the vector field factor TM with the covector field factor T^{*}M.

    Finally, note that it doesn't matter if the extra factors are on the left of the vector field factors,
    they could be permuted arbitrarily; the only thing that matters is symplectic_axis.
    """
    validate_symplectic_axis_of_shape(symplectic_axis, V.shape, shape_name='V.shape')
    return np.tensordot(canonical_symplectic_form_abstract(dtype=dtype), V, ((1,), (symplectic_axis,)))

def symplectic_dual_of_vector_field (V, *, dtype:typing.Any=int) -> np.ndarray:
    """
    Returns the symplectic dual of a vector field on a symplectic manifold M expressed in Darboux coordinates.
    V must have shape (2, n_0, ..., n_{k-1}) for some k >= 0 (k == 0 indicates that the position and momentum
    coordinates are simply scalar values), where the first index distinguishes position and momentum.

    If the shape of V is (2,n), i.e. in standard Darboux coordinates for a section of TM, then the return
    value is given by the [pseudocode] expression

        sum(canonical_symplectic_form_abstract[i,j]*V[j,k] for j in range(2))

    where it should be noted that the symplectic_axis of V is 0.
    """
    validate_darboux_coordinates_quantity_or_raise(V, quantity_name='V')
    return symplectic_dual_of_vector_field_with_extra_factors(V, symplectic_axis=0, dtype=dtype)

def symplectic_dual_of_covector_field_with_extra_factors (C, *, symplectic_axis:int, dtype:typing.Any=int) -> np.ndarray:
    """
    This function is exactly analogous to symplectic_dual_of_vector_field_with_extra_factors except that it operates
    on a tensor field C which is a section of

        F \otimes T^{*}M

    and produces a field which is a section of

        F \otimes TM

    in other words, replacing the covector field factor T^{*}M with the vector field factor TM.

    Finally, note that it doesn't matter if the extra factors are on the left of the vector field factors,
    they could be permuted arbitrarily; the only thing that matters is symplectic_axis.
    """
    validate_symplectic_axis_of_shape(symplectic_axis, C.shape, shape_name='C.shape')
    return np.tensordot(canonical_symplectic_form_abstract_inverse(dtype=dtype), C, ((1,), (symplectic_axis,)))

def symplectic_dual_of_covector_field (C, *, dtype:typing.Any=int) -> np.ndarray:
    """
    Returns the symplectic dual of a covector field on a symplectic manifold M expressed in Darboux coordinates.
    C must have shape (2, n_0, ..., n_{k-1}) for some k >= 0 (k == 0 indicates that the position and momentum
    coordinates are simply scalar values), where the first index distinguishes position and momentum.

    If the shape of C is (2,n), i.e. in standard Darboux coordinates for a section of TM, then the return
    value is given by the [pseudocode] expression

        sum(canonical_symplectic_form_abstract_inverse[i,j]*C[j,k] for j in range(2))

    where it should be noted that the symplectic_axis of C is 0.
    """
    validate_darboux_coordinates_quantity_or_raise(C, quantity_name='C')
    return symplectic_dual_of_covector_field_with_extra_factors(C, symplectic_axis=0, dtype=dtype)

def symplectic_gradient_of (f:typing.Any, qp:np.ndarray) -> np.ndarray:
    """
    Returns the symplectic gradient of the function f with respect to Darboux coordinates qp.
    In particular, this is defined to be the symplectic dual of the covector field df.

    The function f may be a tensor quantity, in which case the return value has shape

        np.shape(f) + np.shape(qp)

    and in particular, if f is a scalar function, the return value has shape equal to that of qp
    (this is the standard case).

    There is a choice of convention regarding where a particular negative sign goes (which
    stems from the choice of sign in the definition of the canonical symplectic form).  See
    the documentation for canonical_symplectic_form_abstract for more on this.  The short story is that
    the convention is picked such that the symplectic gradient of function H(qp) gives the
    vector field whose flow equation is Hamilton's equations where H is the Hamiltonian function.
    """
    validate_darboux_coordinates_quantity_or_raise(qp, quantity_name='qp')
    df = vorpy.symbolic.differential(f, qp)
    assert df.shape == np.shape(f) + qp.shape
    return symplectic_dual_of_covector_field_with_extra_factors(df, symplectic_axis=-len(qp.shape))
