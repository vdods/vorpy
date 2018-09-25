"""
Provides functions for doing symbolic [elementary, i.e. non-covariant] tensor calculus in a way
that is numpy-friendly and can be compiled down to Python code and optionally cached on disk
to be re-imported later.
"""

# import dis
import importlib
import itertools
import numpy as np
import os
import re
import sympy
import sys
import traceback

def __is_python_identifier (s):
    # Check that the dirname is a valid module name.
    alphabetical_or_underscore = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
    digits = '0123456789'
    return len(s) > 0 and s[0] in alphabetical_or_underscore and all(c in alphabetical_or_underscore or c in digits for c in s[1:])

def __one_pass_replacement (s, replacement_d):
    # This was taken from http://stackoverflow.com/questions/6116978/python-replace-multiple-strings
    escaped_replacement_d = dict((re.escape(k),v) for k,v in replacement_d.items())
    pattern = re.compile('|'.join(escaped_replacement_d.keys()))
    return pattern.sub(lambda m:escaped_replacement_d[re.escape(m.group(0))], s)

def multiindex_iterator (shape, *, melt_1_tuple=False):
    """
    Provides a tuple-valued iterator to iterate over all multi-indices with given shape.
    For example, if shape is (2,3), then the iterated sequence is:

        (0,0), (0,1), (0,2), (1,0), (1,1), (1,2).

    If len(shape) is 1 and melt_1_tuple is True (the default is False), then instead of
    returning single-element tuples (0,), (1,), (2,), ..., (n-1,), it returns the plain-old
    integer sequence

        0, 1, 2, ..., n-1.

    Note that if len(shape) is 0, then the iterable sequence will be a single, empty tuple.
    """
    if len(shape) == 1 and melt_1_tuple:
        return range(shape[0])
    else:
        return itertools.product(*map(range, shape))

def variable (name):
    """
    A convenient frontend to sympy.symbols, except that it escapes commas, so that the single
    variable name can contain a comma.  E.g. 'Y[0,1]'.
    """
    return sympy.symbols(name.replace(',','\\,'))

def tensor (name, shape):
    """
    Returns a tensor (a numpy.ndarray with dtype=object) of symbols with the given base name and shape.  
    """
    return np.array(sympy.symbols(name+'_'+'_'.join('(0:{0})'.format(s) for s in shape))).reshape(shape)

def differential (F, X):
    """
    Computes the differential of the given [possibly tensor-valued] symbolic function F with respect to the
    [possibly tensor-valued] symbolic argument X.  If the shape of F is (f_1, f_2, ..., f_m,) and the shape of
    X is (x_1, x_2, ..., x_n,), then the shape of the return value is (f_1, f_2, ..., f_m, x_1, x_2, ..., x_n,).

    Note that the shape of a scalar is (), i.e. the empty tuple.  Thus this function can be used seamlessly
    to compute the differential of any function, scalar- or tensor-valued, and get the expected result.

    For an example of a scalar valued function of 3 variables, if

        X = numpy.array([x,y,z], dtype=object)     # where x,y,z are each symbolic vars

    and

        F = z*sympy.cos(x) - sympy.sin(y**2)/z  # so that F depends on [the components of] X

    then differential(F,X) is

        numpy.array([-z*sympy.sin(x), -2*y*sympy.cos(y**2)/z, sympy.cos(x) + sympy.sin(y**2)/z**2], dtype=object)

    noting that it has shape (3,), which is the concatenation of the shape of F (which is ()) and
    the shape of X (which is (3,)).
    """

    m = len(np.shape(F))
    n = len(np.shape(X))

    # TODO: clean this up

    # Scalar function
    if m == 0:
        # Univariate derivative
        if n == 0:
            return F.diff(X)
        # Multivariate derivative
        else:
            return np.array([F.diff(X[I]) for I in multiindex_iterator(np.shape(X))]).reshape(np.shape(X))

    # Multivariate function
    else:
        # Univariate derivative
        if n == 0:
            return np.array([F[I].diff(X) for I in multiindex_iterator(np.shape(F))]).reshape(np.shape(F))
        # Multivariate derivative
        else:
            retval_shape = tuple(list(np.shape(F))+list(np.shape(X)))
            return np.array([F[I[:m]].diff(X[I[m:]]) for I in multiindex_iterator(retval_shape)]).reshape(retval_shape)

def D (F, *X_v):
    """
    Computes the iterated differential of F with respect to the ordered elements of the iterable X_v.

    For example, if

        X = numpy.array([x,y], dtype=object)        # where x,y are each symbolic vars

    and

        F = x**2 * y + sympy.cos(x)*sympy.sin(y)    # so that F depends on [the components of] X

    then D(F, X, X) is

        numpy.array([
            [2*y - sympy.sin(y)*sympy.cos(x), 2*x - sympy.sin(x)*sympy.cos(y)],
            [2*x - sympy.sin(x)*sympy.cos(y), -sympy.sin(y)*sympy.cos(x)]
        ], dtype=object)

    i.e. the Hessian of F.
    """
    compiled_function = F
    for X in X_v:
        compiled_function = differential(compiled_function, X)
    return compiled_function

def python_expression_for (F, X, *, replacement_d={}, argument_id='X'):
    """
    Return a Python expression for the sybolic m-tensor function F, with respect to the n-tensor variable X.
    Both F and X can be of type numpy.ndarray.  The length of np.shape(F) is m, whereas the length of np.shape(X)
    is n.  F and X can still be scalars as well, they don't have to be tensors.
    """

    m = len(np.shape(F))
    n = len(np.shape(X))

    if n == 0:
        argument = variable(argument_id)
        subs_v = ((X,argument),)
    else:
        argument = np.array(tuple(variable('{0}[{1}]'.format(argument_id, ','.join(map(str,I)))) for I in multiindex_iterator(np.shape(X)))).reshape(np.shape(X))
        subs_v = tuple((X[I],argument[I]) for I in multiindex_iterator(np.shape(X)))

    if m == 0:
        expression_string = repr(F.subs(subs_v))
    else:
        make_substitutions = np.vectorize(lambda expr:expr.subs(subs_v))
        expression_string = repr(make_substitutions(F))

    return __one_pass_replacement(expression_string, replacement_d)

def python_procedure_for (F, X, *, replacement_d={}, argument_id='X', tab_string='    '):
    """
    Return a Python procedure for constructing the sybolic m-tensor function F, with respect to the n-tensor variable X.
    Both F and X can be of type numpy.ndarray.  The length of np.shape(F) is m, whereas the length of np.shape(X)
    is n.  F and X can still be scalars as well, they don't have to be tensors.
    """

    m = len(np.shape(F))
    n = len(np.shape(X))

    if n == 0:
        argument = variable(argument_id)
        subs_v = ((X,argument),)
    else:
        argument = np.array(tuple(variable('{0}[{1}]'.format(argument_id, ','.join(map(str,I)))) for I in multiindex_iterator(np.shape(X)))).reshape(np.shape(X))
        subs_v = tuple((X[I],argument[I]) for I in multiindex_iterator(np.shape(X)))

    if m == 0:
        procedure_string =  tab_string
        procedure_string += 'return '
        procedure_string += repr(F.subs(subs_v))
    else:
        procedure_string =  tab_string
        procedure_string += 'retval = ndarray({0}, dtype=object)\n'.format(F.shape)
        for I in multiindex_iterator(np.shape(F)):
            procedure_string += tab_string
            procedure_string += 'retval[{0}] = {1}\n'.format(','.join(map(str,I)), repr(F[I].subs(subs_v)))
        procedure_string += tab_string
        procedure_string += 'return retval'

    return __one_pass_replacement(procedure_string, replacement_d)

def python_lambda_expression_for (F, X, *, replacement_d={}):
    """
    Return source code for a Python implementation of the sybolic m-tensor function F, with respect to the n-tensor
    variable X.  Both F and X can be of type numpy.ndarray.  The length of np.shape(F) is m, whereas the length of
    np.shape(X) is n.  F and X can still be scalars as well, they don't have to be tensors.
    """
    return 'lambda X:'+python_expression_for(F, X, replacement_d=replacement_d, argument_id='X')

def python_source_code_for (function_id, F, X, *, replacement_d={}, argument_id='X', import_v=[], decorator_v=[]):
    """
    Returns a string containing a free-standing module which defines the given function.  The function_id
    parameter defines the name of the function in the source code.  The import_v and decorator_v parameters
    provide the means to add imports to this source code and decorate the function, respectively.

    See the documentation for python_procedure_for for details on the other parameters.
    """
    retval = ''
    for import_ in import_v:
        retval += import_
        retval += '\n'
    retval += '\n'
    for decorator in decorator_v:
        retval += decorator
        retval += '\n'
    retval += 'def {0} ({1}):\n'.format(function_id, argument_id)
    retval += python_procedure_for(F, X, replacement_d=replacement_d, argument_id=argument_id)
    retval += '\n'
    retval += '\n'
    return retval

def lambdified (F, X, *, replacement_d={}, verbose=False):
    """
    Return a Python function version of the sybolic m-tensor function F, with respect to the n-tensor variable X.
    Both F and X can be of type numpy.ndarray.  The length of np.shape(F) is m, whereas the length of np.shape(X) is n.
    F and X can still be scalars as well, they don't have to be tensors.

    This uses eval to generate the code, and the repr of various things like np.array or sympy.cos show up as
    just 'array' and 'cos', so unless you've imported the correct versions of those into your global namespace,
    you'll need to specify what they each map to in replacement_d.  Also, the np.array will have dtype=object
    unless changed explicitly.  For example,

        replacement_d={'array':'np.array', 'ndarray':'np.ndarray', 'dtype=object':'dtype=float', 'cos':'np.cos'}

    Note: this uses eval, so untrusted input should never be sent to this function.
    See http://www.diveintopython3.net/advanced-iterators.html#eval for more info.
    """

    function_source_code = python_lambda_expression_for(F, X, replacement_d=replacement_d)
    if verbose:
        print(function_source_code)
    compiled_function = eval(function_source_code)
    return compiled_function

def cached_lambdified (function_id, *, function_creator, cache_dirname='lambdified_cache', verbose=False):
    """
    This function has the same purpose as lambdified, except that it attempts to import the function from
    a previously cached source file instead of compiling it fresh each time.  If this attempt fails, it
    calls function_creator, which must return a tuple

        F, X, replacement_d, argument_id, import_v, decorator_v

    which, along with the function_id parameter, specify the arguments to be passed into python_source_code_for.

    cached_lambdified is intended to be used e.g. if it takes a long time to create the symbolic function,
    say due to calls to simplify, or because of a lengthy derivation process such as computing the Hessian
    of a complicated symbolic expression.

    See the documentation for lambdified for more info on the replacement_d argument, and see the documentation
    for python_source_code_for for more info on the other parameters.
    """

    # TODO: make force_recache option

    assert __is_python_identifier(function_id)
    assert __is_python_identifier(cache_dirname)

    function_module_name = cache_dirname + '.' + function_id

    try:
        if verbose:
            print('symbolic.cached_lambdified(): attempting to load module "{0}"...'.format(function_module_name))
        function_module = importlib.import_module(function_module_name)
        if verbose:
            print('symbolic.cached_lambdified(): successfully loaded module "{0}".'.format(function_module_name))
    except:
        if verbose:
            print('symbolic.cached_lambdified(): failed to load module "{0}".'.format(function_module_name))
        if not os.path.exists(cache_dirname):
            if verbose:
                print('symbolic.cached_lambdified(): creating cache directory "{0}".'.format(cache_dirname))
            os.mkdir(cache_dirname)

        cached_function_implementation_filename = os.path.join(cache_dirname,function_id+'.py')
        if verbose:
            print('symbolic.cached_lambdified(): opening cache implementation file "{0}"...'.format(cached_function_implementation_filename))
        try:
            with open(cached_function_implementation_filename, 'wt') as cached_function_implementation_py:
                if verbose:
                    print('symbolic.cached_lambdified(): successfully opened file "{0}".'.format(cached_function_implementation_filename))
                    print('symbolic.cached_lambdified(): generating symbolic function...')
                F,X,replacement_d,argument_id,import_v,decorator_v = function_creator()
                if verbose:
                    print('symbolic.cached_lambdified(): writing source code...')
                cached_function_implementation_py.write(python_source_code_for(function_id, F, X, replacement_d=replacement_d, argument_id=argument_id, import_v=import_v, decorator_v=decorator_v))
                if verbose:
                    print('symbolic.cached_lambdified(): done writing source code.')
            if verbose:
                print('symbolic.cached_lambdified(): attempting to load module "{0}"...'.format(function_module_name))
            # Because new source code was added to the module source tree, invalidate_caches must be called to notify the module system of the update.
            importlib.invalidate_caches()
            function_module = importlib.import_module(cache_dirname + '.' + function_id)
            if verbose:
                print('symbolic.cached_lambdified(): successfully loaded module "{0}".'.format(function_module_name))
        except Exception as e:
            # TODO: handle specific exceptions
            if verbose:
                print('symbolic.cached_lambdified(): error encountered while attempting to cache and load module.  exception {0} was: {1}'.format(type(e), e))
                # Print out the exception traceback (i.e. stack)
                ex_type,ex,tb = sys.exc_info()
                traceback.print_tb(tb)
                print('symbolic.cached_lambdified(): deleting incomplete source file "{0}".'.format(cached_function_implementation_filename))
            os.remove(cached_function_implementation_filename)
            raise

    return function_module.__dict__[function_id]

def symbolic_polynomial (coefficient_prefix, degree, X):
    """
    Returns a generic polynomial of the given degree with symbolic coefficients,
    as well as a list of the coefficients.  X is the coordinates to express the
    polynomial in.  Each polynomial term does not include multiplicity (e.g.
    the `x*y` term would appear as `a_0_1*x*y`, not as `2*a_0_1*x*y`).

    The return value is polynomial, coefficients.

    NOTE: This is not a very efficient implementation; in particular, O(n^degree), where n is the
    dimension of the monomial vector X.  It could be more efficient if only the non-redundant terms
    were computed, and the formula to derive the multiplicity for each term was used.
    """
    # TODO: Allow specification of which degrees should be present in this polynomial

    X_reshaped = X.reshape(-1)

    coefficient_accumulator = []
    polynomial_accumulator = sp.Integer(0)
    for p in range(degree+1):
        degree_shape                = (X_reshaped.size,)*p
        degree_p_coefficients       = vorpy.symbolic.tensor(coefficient_prefix, degree_shape)
        # TODO: Have to encode the symmetries in the coefficients -- in particular, could replace any
        # coefficient with non-strictly-increasing indices with the corresponding one that has
        # strictly increasing indices.
        for I in vorpy.tensor.multiindex_iterator(degree_shape):
            # Replace the non-strictly-increasing-indexed coefficients with 0, and store the rest for return.
            if I != tuple(sorted(I)):
                degree_p_coefficients[I] = 0
            else:
                coefficient_accumulator.append(degree_p_coefficients[I])

        degree_p_variable_tensor    = tensor_power_of_vector(X_reshaped, p)
        # Because of the sparsification done above, multiplying it out this way is somewhat inefficient, but it's fine for now.
        polynomial_accumulator     += np.dot(degree_p_coefficients.reshape(-1), degree_p_variable_tensor.reshape(-1))

    return polynomial_accumulator, coefficient_accumulator
