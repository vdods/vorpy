import itertools
import numpy as np
import vorpy

def test__apply_along_axes__compare_with__apply_along_axis ():
    rng = np.random.RandomState(42)
    a = rng.randn(4,5,6,7)

    assert np.all(vorpy.apply_along_axes(np.sum, [0], [a]) == np.apply_along_axis(np.sum, 0, a))
    assert np.all(vorpy.apply_along_axes(np.sum, [1], [a]) == np.apply_along_axis(np.sum, 1, a))
    assert np.all(vorpy.apply_along_axes(np.sum, [2], [a]) == np.apply_along_axis(np.sum, 2, a))
    assert np.all(vorpy.apply_along_axes(np.sum, [3], [a]) == np.apply_along_axis(np.sum, 3, a))

    assert np.all(vorpy.apply_along_axes(np.sum, (0,), (a,)) == np.apply_along_axis(np.sum, 0, a))
    assert np.all(vorpy.apply_along_axes(np.sum, (1,), (a,)) == np.apply_along_axis(np.sum, 1, a))
    assert np.all(vorpy.apply_along_axes(np.sum, (2,), (a,)) == np.apply_along_axis(np.sum, 2, a))
    assert np.all(vorpy.apply_along_axes(np.sum, (3,), (a,)) == np.apply_along_axis(np.sum, 3, a))

    print('test__apply_along_axes__compare_with__apply_along_axis passed.')

def test__apply_along_axes ():
    def symmetric_square (m):
        return np.einsum('ij,kj->ik', m, m)

    def is_symmetric (m):
        return np.all(m == m.T)

    rng = np.random.RandomState(42)
    a = rng.randn(2,3,4,5)
    N = len(a.shape)

    # Use all possible combinations of input and output axes.
    for input_i0 in range(-N,N-1):
        for input_i1 in range(input_i0+1,N):
            # Only test the pairs where input_i0 indexes an axis before that indexed by input_i1.
            if input_i0 + N >= input_i1:
                continue

            for output_i0 in range(-N,N-1):
                for output_i1 in range(output_i0+1,N):
                    # Only test the pairs where output_i0 indexes an axis before that indexed by output_i1.
                    if output_i0 + N >= output_i1:
                        continue

                    output_axis_v = (output_i0,output_i1)
                    # Compute the result.  The multi-slice across the output axes should be a symmetric matrix.
                    result = vorpy.apply_along_axes(symmetric_square, (input_i0,input_i1), (a,), output_axis_v=output_axis_v)
                    # Figure out which indices correspond to the input axes; call these result_non_output_axis_v.
                    all_indices = tuple(range(N))
                    normalized_output_axis_v = tuple(output_axis if output_axis >= 0 else output_axis+N for output_axis in output_axis_v)
                    result_non_output_axis_v = sorted(list(frozenset(all_indices) - frozenset(normalized_output_axis_v)))
                    # print('output_axis_v = {0}, result_non_output_axis_v = {1}'.format(output_axis_v, result_non_output_axis_v))
                    assert len(result_non_output_axis_v) == 2
                    # Take all multi-slices and verify symmetry.
                    for check_i0,check_i1 in itertools.product(range(result.shape[result_non_output_axis_v[0]]), range(result.shape[result_non_output_axis_v[1]])):
                        # Construct the multi-slice to take.
                        multislice = [slice(None) for _ in range(4)]
                        multislice[result_non_output_axis_v[0]] = check_i0
                        multislice[result_non_output_axis_v[1]] = check_i1
                        # print('multislice = {0}'.format(multislice))
                        assert is_symmetric(result[tuple(multislice)])

    print('test__apply_along_axes passed.')

