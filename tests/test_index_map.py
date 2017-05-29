import numpy as np
import vorpy

def test__is_subsequence_of_nondecreasing_sequence ():
    # Positive tests

    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([], [])
    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([], [0])
    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([], [0,1])
    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([], [0,1,2])

    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([0], [0])
    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([0], [0,1])
    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([0], [0,1,2])

    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([1], [0,1])
    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([1], [0,1,2])

    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([2], [0,1,2])

    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([0,1], [0,1])
    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([0,1], [0,1,2])

    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([0,2], [0,1,2])

    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([1,2], [0,1,2])

    assert vorpy.index_map.is_subsequence_of_nondecreasing_sequence([0,1,2], [0,1,2])

    # Negative tests

    assert not vorpy.index_map.is_subsequence_of_nondecreasing_sequence([0], [])
    assert not vorpy.index_map.is_subsequence_of_nondecreasing_sequence([0], [1])
    assert not vorpy.index_map.is_subsequence_of_nondecreasing_sequence([1,0], [1])
    assert not vorpy.index_map.is_subsequence_of_nondecreasing_sequence([0,1,2], [0,1])
    assert not vorpy.index_map.is_subsequence_of_nondecreasing_sequence([0,1,1], [0,1,2])
    assert not vorpy.index_map.is_subsequence_of_nondecreasing_sequence([0,1,3], [0,1,2])
    assert not vorpy.index_map.is_subsequence_of_nondecreasing_sequence([0,4,2], [0,1,2])
    assert not vorpy.index_map.is_subsequence_of_nondecreasing_sequence([5,1,2], [0,1,2])
    assert not vorpy.index_map.is_subsequence_of_nondecreasing_sequence([5,6,2], [0,1,2])
    assert not vorpy.index_map.is_subsequence_of_nondecreasing_sequence([5,6,7], [0,1,2])
    assert not vorpy.index_map.is_subsequence_of_nondecreasing_sequence([5,6,7], [0,1])
    assert not vorpy.index_map.is_subsequence_of_nondecreasing_sequence([5,6,7], [0])
    assert not vorpy.index_map.is_subsequence_of_nondecreasing_sequence([5,6,7], [])

    print('test__is_subsequence_of_nondecreasing_sequence passed.')

# This shouldn't be run directly -- see setting of __test__ attribute below.
def test__index_map_inverse__case (index_map_v, index_count):
    index_map_inverse_v = vorpy.index_map.index_map_inverse(index_map_v, index_count)
    # print('')
    # print('index_map_v = {0}, index_count = {1}'.format(index_map_v, index_count))
    # print('index_map_inverse_v = {0}'.format(index_map_inverse_v))
    # print('np.arange(len(index_map_v)) = {0}'.format(np.arange(len(index_map_v))))
    # print('compose_index_maps(index_map_inverse_v, index_map_v) = {0}'.format(compose_index_maps(index_map_inverse_v, index_map_v)))
    # print('compose_index_maps(index_map_v, index_map_inverse_v) = {0}'.format(compose_index_maps(index_map_v, index_map_inverse_v)))
    assert np.all(vorpy.index_map.compose_index_maps(index_map_inverse_v, index_map_v) == np.arange(len(index_map_v)))

# Indicate to nose not to run this.  TODO: Change testMatch to be more reasonable.
test__index_map_inverse__case.__test__ = False

def test__index_map_inverse ():
    test__index_map_inverse__case([], 0)

    test__index_map_inverse__case([], 1)
    test__index_map_inverse__case([0], 1)

    test__index_map_inverse__case([], 2)
    test__index_map_inverse__case([0], 2)
    test__index_map_inverse__case([1], 2)
    test__index_map_inverse__case([0,1], 2)
    test__index_map_inverse__case([1,0], 2)

    test__index_map_inverse__case([], 3)
    test__index_map_inverse__case([0], 3)
    test__index_map_inverse__case([1], 3)
    test__index_map_inverse__case([2], 3)
    test__index_map_inverse__case([0,1], 3)
    test__index_map_inverse__case([1,0], 3)
    test__index_map_inverse__case([0,2], 3)
    test__index_map_inverse__case([2,0], 3)
    test__index_map_inverse__case([1,2], 3)
    test__index_map_inverse__case([2,1], 3)
    test__index_map_inverse__case([0,1,2], 3)
    test__index_map_inverse__case([0,2,1], 3)
    test__index_map_inverse__case([1,0,2], 3)
    test__index_map_inverse__case([1,2,0], 3)
    test__index_map_inverse__case([2,0,1], 3)
    test__index_map_inverse__case([2,1,0], 3)

    print('test__index_map_inverse passed.')
