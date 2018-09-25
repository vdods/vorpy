import numpy as np
import sympy as sp
import sys
import traceback
import vorpy.symbolic
import vorpy.tensor

def test_contract ():
    # Define a bunch of tensors to use in the tests
    x = sp.symbols('x')
    T_ = vorpy.symbolic.tensor('z', tuple())
    T_4 = vorpy.symbolic.tensor('a', (4,))
    T_5 = vorpy.symbolic.tensor('b', (5,))
    U_5 = vorpy.symbolic.tensor('c', (5,))
    T_3_5 = vorpy.symbolic.tensor('d', (3,5))
    T_4_3 = vorpy.symbolic.tensor('e', (4,3))
    T_4_4 = vorpy.symbolic.tensor('f', (4,4))
    T_5_2 = vorpy.symbolic.tensor('g', (5,2))
    T_3_4_5 = vorpy.symbolic.tensor('h', (3,4,5))
    T_3_3_4 = vorpy.symbolic.tensor('i', (3,3,4))

    def is_zero_tensor (T):
        return all(t == 0 for t in T.flat) if hasattr(T,'shape') else (T == 0)

    def positive__unit_test_0a ():
        output_shape = (3,5,3)
        contraction_shape = (4,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_3_4_5[i,j,k]*T_4_3[j,l] for (j,) in vorpy.tensor.multiindex_iterator(contraction_shape)) for i,k,l in vorpy.tensor.multiindex_iterator(output_shape)]))
        assert is_zero_tensor(vorpy.tensor.contract('ijk,jl', T_3_4_5, T_4_3, dtype=object) - expected_result)
    def positive__unit_test_0b ():
        output_shape = (3,5,3)
        contraction_shape = (4,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_3_4_5[i,j,k]*T_4_3[j,l] for (j,) in vorpy.tensor.multiindex_iterator(contraction_shape)) for i,k,l in vorpy.tensor.multiindex_iterator(output_shape)]))
        assert is_zero_tensor(vorpy.tensor.contract('ijk,jl', T_3_4_5, T_4_3, output='ikl', dtype=object) - expected_result)
    def positive__unit_test_0c ():
        output_shape = (3,3,5)
        contraction_shape = (4,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_3_4_5[i,j,k]*T_4_3[j,l] for (j,) in vorpy.tensor.multiindex_iterator(contraction_shape)) for i,l,k in vorpy.tensor.multiindex_iterator(output_shape)]))
        assert is_zero_tensor(vorpy.tensor.contract('ijk,jl', T_3_4_5, T_4_3, output='ilk', dtype=object) - expected_result)

    def positive__unit_test_1a ():
        output_shape = (5,)
        contraction_shape = (3,4)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_3_4_5[i,j,k]*T_4_3[j,i] for i,j in vorpy.tensor.multiindex_iterator(contraction_shape)) for (k,) in vorpy.tensor.multiindex_iterator(output_shape)]))
        assert is_zero_tensor(vorpy.tensor.contract('ijk,ji', T_3_4_5, T_4_3, dtype=object) - expected_result)
    def positive__unit_test_1b ():
        output_shape = (5,)
        contraction_shape = (3,4)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_3_4_5[i,j,k]*T_4_3[j,i] for i,j in vorpy.tensor.multiindex_iterator(contraction_shape)) for (k,) in vorpy.tensor.multiindex_iterator(output_shape)]))
        assert is_zero_tensor(vorpy.tensor.contract('ijk,ji', T_3_4_5, T_4_3, output='k', dtype=object) - expected_result)

    def positive__unit_test_2a ():
        output_shape = tuple()
        contraction_shape = (5,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_5[i]*T_5[i] for (i,) in vorpy.tensor.multiindex_iterator(contraction_shape))]))
        assert is_zero_tensor(vorpy.tensor.contract('i,i', T_5, T_5, dtype=object) - expected_result)
    def positive__unit_test_2b ():
        output_shape = tuple()
        contraction_shape = (5,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_5[i]*T_5[i] for (i,) in vorpy.tensor.multiindex_iterator(contraction_shape))]))
        assert is_zero_tensor(vorpy.tensor.contract('i,i', T_5, T_5, output='', dtype=object) - expected_result)

    def positive__unit_test_3a ():
        output_shape = (5,5)
        contraction_shape = tuple()
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([T_5[i]*U_5[j] for i,j in vorpy.tensor.multiindex_iterator(output_shape)]))
        assert is_zero_tensor(vorpy.tensor.contract('i,j', T_5, U_5, dtype=object) - expected_result)
    def positive__unit_test_3b ():
        output_shape = (5,5)
        contraction_shape = tuple()
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([T_5[j]*U_5[i] for i,j in vorpy.tensor.multiindex_iterator(output_shape)]))
        assert is_zero_tensor(vorpy.tensor.contract('j,i', T_5, U_5, dtype=object) - expected_result)
    def positive__unit_test_3c ():
        output_shape = (5,5)
        contraction_shape = tuple()
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([T_5[i]*U_5[j] for i,j in vorpy.tensor.multiindex_iterator(output_shape)]))
        assert is_zero_tensor(vorpy.tensor.contract('i,j', T_5, U_5, output='ij', dtype=object) - expected_result)
    def positive__unit_test_3d ():
        output_shape = (5,5)
        contraction_shape = tuple()
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([T_5[i]*U_5[j] for j,i in vorpy.tensor.multiindex_iterator(output_shape)]))
        assert is_zero_tensor(vorpy.tensor.contract('i,j', T_5, U_5, output='ji', dtype=object) - expected_result)

    def positive__unit_test_4a ():
        output_shape = (4,2)
        contraction_shape = (3,5)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_4_3[i,j]*T_3_5[j,k]*T_5_2[k,l] for j,k in vorpy.tensor.multiindex_iterator(contraction_shape)) for i,l in vorpy.tensor.multiindex_iterator(output_shape)]))
        assert is_zero_tensor(vorpy.tensor.contract('ij,jk,kl', T_4_3, T_3_5, T_5_2, dtype=object) - expected_result)
    def positive__unit_test_4b ():
        output_shape = (2,4)
        contraction_shape = (3,5)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_4_3[i,j]*T_3_5[j,k]*T_5_2[k,l] for j,k in vorpy.tensor.multiindex_iterator(contraction_shape)) for l,i in vorpy.tensor.multiindex_iterator(output_shape)]))
        assert is_zero_tensor(vorpy.tensor.contract('lj,jk,ki', T_4_3, T_3_5, T_5_2, dtype=object) - expected_result)
    def positive__unit_test_4c ():
        output_shape = (4,2)
        contraction_shape = (3,5)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_4_3[i,j]*T_3_5[j,k]*T_5_2[k,l] for j,k in vorpy.tensor.multiindex_iterator(contraction_shape)) for i,l in vorpy.tensor.multiindex_iterator(output_shape)]))
        assert is_zero_tensor(vorpy.tensor.contract('ij,jk,kl', T_4_3, T_3_5, T_5_2, output='il', dtype=object) - expected_result)

    def positive__unit_test_5a ():
        output_shape = tuple()
        contraction_shape = (4,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_4_4[i,i] for (i,) in vorpy.tensor.multiindex_iterator(contraction_shape))]))
        assert is_zero_tensor(vorpy.tensor.contract('ii', T_4_4, dtype=object) - expected_result)
    def positive__unit_test_5b ():
        output_shape = tuple()
        contraction_shape = (4,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_4_4[i,i] for (i,) in vorpy.tensor.multiindex_iterator(contraction_shape))]))
        assert is_zero_tensor(vorpy.tensor.contract('ii', T_4_4, output='', dtype=object) - expected_result)

    def positive__unit_test_6a ():
        output_shape = (4,)
        contraction_shape = (3,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_3_3_4[i,i,j] for (i,) in vorpy.tensor.multiindex_iterator(contraction_shape)) for (j,) in vorpy.tensor.multiindex_iterator(output_shape)]))
        assert is_zero_tensor(vorpy.tensor.contract('iij', T_3_3_4, dtype=object) - expected_result)
    def positive__unit_test_6b ():
        output_shape = (4,)
        contraction_shape = (3,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_3_3_4[i,i,j] for (i,) in vorpy.tensor.multiindex_iterator(contraction_shape)) for (j,) in vorpy.tensor.multiindex_iterator(output_shape)]))
        assert is_zero_tensor(vorpy.tensor.contract('iij', T_3_3_4, output='j', dtype=object) - expected_result)

    def positive__unit_test_7a ():
        expected_result = T_*T_
        assert is_zero_tensor(vorpy.tensor.contract(',', T_, T_, dtype=object) - expected_result)
    def positive__unit_test_7b ():
        expected_result = T_*x
        assert is_zero_tensor(vorpy.tensor.contract(',', T_, x, dtype=object) - expected_result)
    def positive__unit_test_7c ():
        expected_result = T_*x
        assert is_zero_tensor(vorpy.tensor.contract(',', x, T_, dtype=object) - expected_result)
    def positive__unit_test_7d ():
        expected_result = x*x
        assert is_zero_tensor(vorpy.tensor.contract(',', x, x, dtype=object) - expected_result)

    def positive__unit_test_8a ():
        assert is_zero_tensor(vorpy.tensor.contract('', T_, dtype=object) - T_)
    def positive__unit_test_8b ():
        assert is_zero_tensor(vorpy.tensor.contract('', x, dtype=object) - x)

    def positive__unit_test_9a ():
        # We will allow summation over indices that occur more than twice, even though
        # this indicates a type error in tensorial constructions.  But here, we're just
        # working with tensor-like grids of values, so no such assumption will be made.
        # Perhaps a warning could be printed, which could be turned off by the explicit
        # specification of a keyword argument.
        output_shape = tuple()
        contraction_shape = (5,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_5[i]*T_5[i]*U_5[i] for (i,) in vorpy.tensor.multiindex_iterator(contraction_shape))]))
        assert is_zero_tensor(vorpy.tensor.contract('i,i,i', T_5, T_5, U_5, dtype=object) - expected_result)

    def negative__unit_test_0a ():
        vorpy.tensor.contract('', T_5, T_4_4, dtype=object) # Wrong number of index strings.
    def negative__unit_test_0b ():
        vorpy.tensor.contract('i,j,k', T_5, T_4_4, dtype=object) # Wrong number of index strings.
    def negative__unit_test_0c ():
        vorpy.tensor.contract('i,j,k', T_4_4, dtype=object) # Wrong number of index strings.
    def negative__unit_test_0d ():
        vorpy.tensor.contract('i,j', dtype=object) # Wrong number of index strings.

    def negative__unit_test_1a ():
        vorpy.tensor.contract('', T_5, dtype=object) # Mismatch of number of indices and tensor order.
    def negative__unit_test_1b ():
        vorpy.tensor.contract('ij', T_5, dtype=object) # Mismatch of number of indices and tensor order.
    def negative__unit_test_1c ():
        vorpy.tensor.contract('ij', T_3_4_5, dtype=object) # Mismatch of number of indices and tensor order.

    def negative__unit_test_2a ():
        vorpy.tensor.contract('i,i', T_5, T_4, dtype=object) # Non-matching contraction dimensions.
    def negative__unit_test_2b ():
        vorpy.tensor.contract('i,i,i', T_5, T_4, T_4, dtype=object) # Non-matching contraction dimensions.
    def negative__unit_test_2c ():
        vorpy.tensor.contract('ij,jk', T_4_3, T_4_4, dtype=object) # Non-matching contraction dimensions.
    def negative__unit_test_2d ():
        vorpy.tensor.contract('ij,ij', T_4_3, T_4_4, dtype=object) # Non-matching contraction dimensions.
    def negative__unit_test_2e ():
        vorpy.tensor.contract('ij,ij', T_5_2, T_4_4, dtype=object) # Non-matching contraction dimensions.

    def negative__unit_test_3a ():
        vorpy.tensor.contract('ij,jk', T_4_3, T_3_5, output='ii', dtype=object)

    # Run all unit tests in alphabetical order.  The set of unit tests is defined
    # to be the set of callable local objects (see locals()), where an object obj is
    # callable iff hasattr(obj,'__call__') returns True.
    unit_test_count = 0
    pass_count = 0
    fail_count = 0
    for name in sorted(locals().keys()):
        obj = locals()[name]
        if hasattr(obj,'__call__'):
            # Positive and negative tests are run differently.
            if 'positive' in name:
                assert 'negative' not in name, 'Exactly one of the strings \'positive\' and \'negative\' should be present in a unit test name (in particular, the failing name is \'{0}\').'.format(name)
                unit_test_count += 1
                sys.stdout.write('Running {0} ... '.format(name))
                try:
                    obj()
                    sys.stdout.write('passed (no exception was raised).\n')
                    pass_count += 1
                except Exception as e:
                    sys.stdout.write('FAILED -- exception was {0}, stack trace was\n{1}\n'.format(repr(e), traceback.format_exc()))
                    fail_count += 1
            elif 'negative' in name:
                assert 'positive' not in name, 'Exactly one of the strings \'positive\' and \'negative\' should be present in a unit test name (in particular, the failing name is \'{0}\').'.format(name)
                unit_test_count += 1
                sys.stdout.write('Running {0} ... '.format(name))
                try:
                    obj() # In a negative test, we expect an exception to be raised.
                    sys.stdout.write('FAILED (expected exception to be raised in negative test, but none was raised).\n')
                    fail_count += 1
                except Exception as e:
                    sys.stdout.write('passed (caught expected exception {0}).\n'.format(repr(e)))
                    pass_count += 1
    if unit_test_count > 0:
        print('Summary: {0} unit tests, {1} passed, {2} failed, failure rate was {3}%'.format(unit_test_count, pass_count, fail_count, float(fail_count)*100.0/unit_test_count))

def tensor_power_of_vector__test ():
    V = np.array((sp.var('x'), sp.var('y'), sp.var('z')))

    #print(f'V = {V}')
    #for p in range(5):
        #print(f'vorpy.tensor.tensor_power_of_vector(V, {p}):')
        #print(f'{vorpy.tensor.tensor_power_of_vector(V, p)}')
        #print()

    # Specific comparisons

    power           = 0
    expected_value  = 1
    actual_value    = vorpy.tensor.tensor_power_of_vector(V, power)
    if not np.all(expected_value == actual_value):
        raise FancyException(f'For power {power}, expected {expected_value} but actual value was {actual_value}')

    power           = 1
    expected_value  = V
    actual_value    = vorpy.tensor.tensor_power_of_vector(V, power)
    if not np.all(expected_value == actual_value):
        raise FancyException(f'For power {power}, expected {expected_value} but actual value was {actual_value}')

    power           = 2
    expected_value  = vorpy.tensor.contract('i,j', V, V, dtype=object)
    actual_value    = vorpy.tensor.tensor_power_of_vector(V, power)
    if not np.all(expected_value == actual_value):
        raise FancyException(f'For power {power}, expected {expected_value} but actual value was {actual_value}')

    power           = 3
    expected_value  = vorpy.tensor.contract('i,j,k', V, V, V, dtype=object)
    actual_value    = vorpy.tensor.tensor_power_of_vector(V, power)
    if not np.all(expected_value == actual_value):
        raise FancyException(f'For power {power}, expected {expected_value} but actual value was {actual_value}')

    power           = 4
    expected_value  = vorpy.tensor.contract('i,j,k,l', V, V, V, V, dtype=object)
    actual_value    = vorpy.tensor.tensor_power_of_vector(V, power)
    if not np.all(expected_value == actual_value):
        raise FancyException(f'For power {power}, expected {expected_value} but actual value was {actual_value}')

    print('tensor_power_of_vector__test passed')
