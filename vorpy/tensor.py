"""
Provides functions for doing general tensor operations, notably tensor contraction.
In particular, vorpy.tensor.contract works like numpy.einsum but works on dtype=object
(but is probably MUCH slower).
"""

import itertools
import numpy as np
import operator
import vorpy

class VorpyTensorException(vorpy.VorpyException):
    """
    Base class for all exceptions generated by the vorpy.tensor module (this doesn't
    necessarily include other exceptions generated by functions called within the
    vorpy.tensor module).
    """
    pass

class VorpyTensorProgrammerError(vorpy.VorpyProgrammerError):
    """Base class for all internal programmer error exceptions generated by the vorpy.tensor module."""
    pass

def order (T):
    """
    Returns the tensor order of T.  The order of a tensor is equivalent to the number of indices
    necessary to address one of its components.  In particular, the tensor order of a scalar is 0.
    """
    return len(T.shape) if hasattr(T,'shape') else 0

def shape (T):
    """
    Returns the shape of the tensor T.  If T is a scalar, then the shape is tuple().
    Note that multiindex_iterator(shape(T)) can be used to iterate over the components
    of T.  See also component.
    """
    return T.shape if hasattr(T,'shape') else tuple()

def component (T, multiindex):
    """
    Returns the component of T specified by the given multiindex.  In particular, if T has tensor
    order 0 (i.e. is a scalar), then this just returns T itself.
    """
    return T[multiindex] if hasattr(T,'shape') else T

def multiindex_iterator (multiindex_shape):
    """
    Returns an iterator for a multiindex for the given shape.  For example,
    multiindex_iterator(shape(T)) can be used to iterate over the components of T.
    """
    return itertools.product(*tuple(range(dim) for dim in multiindex_shape))

def contract (contraction_string, *tensors, **kwargs):
    """
    This is meant to do the same thing as numpy.einsum, except that it can handle dtype=object
    (but is probably MUCH slower).
    """

    def positions_of_all_occurrences_of_char (s, c):
        for pos,ch in enumerate(s):
            if ch == c:
                yield pos

    output_index_string = kwargs.get('output', None)
    if 'dtype' not in kwargs:
        raise VorpyTensorException('Must specify the \'dtype\' keyword argument (e.g. dtype=float, dtype=object, etc).')
    dtype = kwargs['dtype']
    error_messages = []

    #
    # Starting here is just checking that the contraction is well-defined, such as checking
    # the summation semantics of the contracted and free indices, checking that the contracted
    # slots' dimensions match, etc.
    # 

    # Verify that the indices in the contraction string match the orders of the tensor arguments.
    index_strings = contraction_string.split(',')
    if len(index_strings) != len(tensors):
        raise VorpyTensorException('There must be the same number of comma-delimited index strings (which in this case is {0}) as tensor arguments (which in this case is {1}).'.format(len(index_strings), len(tensors)))
    all_index_counts_matched = True
    for i,(index_string,tensor) in enumerate(zip(index_strings,tensors)):
        if len(index_string) != order(tensor):
            error_messages.append('the number of indices in {0}th index string \'{1}\' (which in this case is {2}) did not match the order of the corresponding tensor argument (which in this case is {3})'.format(i, index_string, len(index_string), order(tensor)))
            all_index_counts_matched = False
    if not all_index_counts_matched:
        raise VorpyTensorException('At least one index string had a number of indices that did not match the order of its corresponding tensor argument.  In particular, {0}.'.format(', '.join(error_messages)))

    # Determine which indices are to be contracted (defined as any indices occurring more than once)
    # and determine the free indices (defined as any indices occurring exactly once).
    indices = frozenset(c for c in contraction_string if c != ',')
    contraction_indices = frozenset(c for c in indices if contraction_string.count(c) > 1)
    free_indices = indices - contraction_indices # Set subtraction    
    
    # If the 'output' keyword argument wasn't specified, use the alphabetization of free_indices
    # as the output indices.
    if output_index_string == None:
        output_indices = free_indices
        output_index_string = ''.join(sorted(list(free_indices)))
    # Otherwise, perform some verification on output_index_string.
    else:
        # If the 'output' keyword argument was specified (stored in output_index_string), 
        # then verify that it's well-defined, in that that output_index_string contains
        # unique characters.
        output_indices = frozenset(output_index_string)
        output_indices_are_unique = True
        for index in output_indices:
            if output_index_string.count(index) > 1:
                error_messages.append('index \'{0}\' occurs more than once'.format(index))
                output_indices_are_unique = False
        if not output_indices_are_unique:
            raise VorpyTensorException('The characters of the output keyword argument (which in this case is \'{0}\') must be unique.  In particular, {1}.'.format(output_index_string, ', '.join(error_messages)))
        # Verify that free_indices and output_index_string contain exactly the same characters.
        if output_indices != free_indices:
            raise VorpyTensorException('The output indices (which in this case are \'{0}\') must be precisely the free indices (which in this case are \'{1}\').'.format(''.join(sorted(output_indices)), ''.join(sorted(free_indices))))

    # Verify that the dimensions of each of contraction_indices match, while constructing
    # an indexed list of the dimensions of the contracted slots.
    contraction_index_string = ''.join(sorted(list(contraction_indices)))
    contracted_indices_dimensions_match = True
    for contraction_index in contraction_index_string:
        indexed_slots_and_dims = []
        for arg_index,(index_string,tensor) in enumerate(zip(index_strings,tensors)):
            for slot_index in positions_of_all_occurrences_of_char(index_string,contraction_index):
                indexed_slots_and_dims.append((arg_index,slot_index,tensor.shape[slot_index]))
        distinct_dims = frozenset(dim for arg_index,slot_index,dim in indexed_slots_and_dims)
        if len(distinct_dims) > 1:
            slot_indices = ','.join('{0}th'.format(slot_index) for _,slot_index,_ in indexed_slots_and_dims)
            arg_indices = ','.join('{0}th'.format(arg_index) for arg_index,_,_ in indexed_slots_and_dims)
            dims = ','.join('{0}'.format(dim) for _,_,dim in indexed_slots_and_dims)
            error_messages.append('index \'{0}\' is used to contract the {1} slots respectively of the {2} tensor arguments whose respective slots have non-matching dimensions {3}'.format(contraction_index, slot_indices, arg_indices, dims))
            contracted_indices_dimensions_match = False
    if not contracted_indices_dimensions_match:
        raise VorpyTensorException('The dimensions of at least one set of contracted tensor slots did not match.  In particular, {0}.'.format(', '.join(error_messages)))

    def dims_of_index_string (index_string):
        def tensor_and_slot_in_which_index_occurs (index):
            for index_string,tensor in zip(index_strings,tensors):
                slot = index_string.find(index)
                if slot >= 0:
                    return tensor,slot
            raise VorpyTensorProgrammerError('This should never happen.')
        lookup = tuple(tensor_and_slot_in_which_index_occurs(index) for index in index_string)
        return tuple(tensor.shape[slot] for tensor,slot in lookup)

    contraction_dims = dims_of_index_string(contraction_index_string)
    output_dims = dims_of_index_string(output_index_string)

    #
    # Starting here is the actual contraction computation
    #

    def component_indices_function (index_string):
        is_contraction_index = tuple(index in contraction_index_string for index in index_string)
        lookups = tuple((0 if is_contraction_index[i] else 1, contraction_index_string.index(index) if is_contraction_index[i] else output_index_string.index(index)) for i,index in enumerate(index_string))

        index_string_pair = (contraction_index_string, output_index_string)
        for i,lookup in enumerate(lookups):
            if index_string[i] != index_string_pair[lookup[0]][lookup[1]]:
                raise VorpyTensorProgrammerError('This should not happen')

        def component_indices_of (contracted_and_output_indices_tuple):
            if len(lookups) != len(index_string):
                raise VorpyTensorProgrammerError('This should not happen')
            if len(contracted_and_output_indices_tuple) != 2:
                raise VorpyTensorProgrammerError('This should not happen')
            if len(contracted_and_output_indices_tuple[0]) != len(contraction_index_string):
                raise VorpyTensorProgrammerError('This should not happen')
            if len(contracted_and_output_indices_tuple[1]) != len(output_index_string):
                raise VorpyTensorProgrammerError('This should not happen')
            retval = tuple(contracted_and_output_indices_tuple[lookup[0]][lookup[1]] for lookup in lookups)
            return retval

        test_output = ''.join(component_indices_of((contraction_index_string, output_index_string)))
        if test_output != index_string:
            raise VorpyTensorProgrammerError('This should not happen')
        return component_indices_of

    component_indices_functions = tuple(component_indices_function(index_string) for index_string in index_strings)

    def product_of_components_of_tensors (contracted_and_output_indices_tuple):
        return reduce(operator.mul, tuple(component(tensor,component_indices_function(contracted_and_output_indices_tuple)) for tensor,component_indices_function in zip(tensors,component_indices_functions)), 1)

    def computed_component (output_component_indices):
        return sum(product_of_components_of_tensors((contraction_component_indices, output_component_indices)) for contraction_component_indices in multiindex_iterator(contraction_dims))

    retval = np.ndarray(output_dims, dtype=dtype, buffer=np.array([computed_component(output_component_indices) for output_component_indices in multiindex_iterator(output_dims)]))
    # If the result is a 0-tensor, then coerce it to the scalar type.
    if retval.shape == tuple():
        retval = retval[tuple()]
    return retval

