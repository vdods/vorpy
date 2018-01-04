import filecmp
import os
import sys
import vorpy.pickle

TEST_ARTIFACTS_DIR = 'test_artifacts/pickle'

def make_filename_in_artifacts_dir (filename):
    os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)
    return os.path.join(TEST_ARTIFACTS_DIR, filename)

def delete_file_if_exists (filename):
    if os.path.exists(filename):
        os.remove(filename)
    assert not os.path.exists(filename)

def assert_files_are_equal (filename0, filename1):
    assert filecmp.cmp(filename0, filename1, shallow=False)

def test__pickle_unpickle ():
    pickle_filename = make_filename_in_artifacts_dir('test__pickle_unpickle.pickle')
    delete_file_if_exists(pickle_filename)

    data = [x**2.5 for x in range(-100,101)]
    vorpy.pickle.pickle(data=data, pickle_filename=pickle_filename, log_out=sys.stdout)

    assert os.path.exists(pickle_filename)
    unpickled_data = vorpy.pickle.unpickle(pickle_filename=pickle_filename, log_out=sys.stdout)
    assert data == unpickled_data

def test__unpickle_pickle ():
    # This test depends on test__pickle_unpickle already having passed.

    source_pickle_filename = make_filename_in_artifacts_dir('test__unpickle_pickle.source.pickle')
    dest_pickle_filename = make_filename_in_artifacts_dir('test__unpickle_pickle.dest.pickle')
    for filename in [source_pickle_filename, dest_pickle_filename]:
        delete_file_if_exists(filename)

    # First, create a pickle file.
    data = ['a', 123, 'b', 456]
    vorpy.pickle.pickle(data=data, pickle_filename=source_pickle_filename, log_out=sys.stdout)
    assert os.path.exists(source_pickle_filename)

    # Now attempt to unpickle and then pickle it.
    unpickled_data = vorpy.pickle.unpickle(pickle_filename=source_pickle_filename, log_out=sys.stdout)
    assert data == unpickled_data
    vorpy.pickle.pickle(data=unpickled_data, pickle_filename=dest_pickle_filename, log_out=sys.stdout)

    # Now ensure that the dest file exists and the source and dest files are identical.
    assert os.path.exists(dest_pickle_filename)
    assert_files_are_equal(source_pickle_filename, dest_pickle_filename)

def test__try_to_pickle ():
    pickle_filename = make_filename_in_artifacts_dir('test__try_to_pickle.pickle')
    delete_file_if_exists(pickle_filename)

    data = [x**2.5 for x in range(-100,101)]
    pickle_succeeded = vorpy.pickle.try_to_pickle(data=data, pickle_filename=pickle_filename, log_out=sys.stdout)
    assert pickle_succeeded
    assert os.path.exists(pickle_filename)

def test__try_to_pickle__lambda ():
    pickle_filename = make_filename_in_artifacts_dir('test__try_to_pickle__lambda.pickle')
    delete_file_if_exists(pickle_filename)

    L = lambda x:x**2
    pickle_succeeded = vorpy.pickle.try_to_pickle(data=L, pickle_filename=pickle_filename, log_out=sys.stdout)
    assert pickle_succeeded
    assert os.path.exists(pickle_filename)

    # Now attempt to unpickle and then pickle it.
    unpickled_L = vorpy.pickle.unpickle(pickle_filename=pickle_filename, log_out=sys.stdout)
    assert all(L(x) == unpickled_L(x) for x in range(100))

def test__try_to_pickle__failure_IOError ():
    # This should produce an error because this is a directory.
    path_to_produce_IOError = make_filename_in_artifacts_dir('test__try_to_pickle__failure_IOError.nonexistent_filename/')

    data = [x**2.5 for x in range(-100,101)]
    pickle_succeeded = vorpy.pickle.try_to_pickle(data=data, pickle_filename=path_to_produce_IOError, log_out=sys.stdout)
    assert not pickle_succeeded

def test__try_to_pickle__failure_PickleError ():
    pickle_filename = make_filename_in_artifacts_dir('test__try_to_pickle.pickle')
    delete_file_if_exists(pickle_filename)

    # Can't pickle a generator.
    data = (x**2 for x in range(100))
    pickle_succeeded = vorpy.pickle.try_to_pickle(data=data, pickle_filename=pickle_filename, log_out=sys.stdout)
    assert not pickle_succeeded
    assert not os.path.exists(pickle_filename)

def test__pickle_with_transform ():
    pickle_filename_expected = make_filename_in_artifacts_dir('test__pickle_with_transform.expected.pickle')
    pickle_filename_actual = make_filename_in_artifacts_dir('test__pickle_with_transform.actual.pickle')
    for filename in [pickle_filename_expected, pickle_filename_actual]:
        delete_file_if_exists(filename)

    original_data = list(range(-100,101))
    transform = lambda data:[x**2.5 for x in data]
    transformed_data = transform(original_data)

    # Pickle the transformed data
    vorpy.pickle.pickle(data=transformed_data, pickle_filename=pickle_filename_expected, log_out=sys.stdout)
    # Have the function do the transform and then pickle.
    vorpy.pickle.pickle(data=original_data, pickle_filename=pickle_filename_actual, transform_before_pickle=transform, log_out=sys.stdout)

    # Ensure the results are the same.
    assert_files_are_equal(pickle_filename_actual, pickle_filename_expected)

def test__unpickle_with_transform ():
    # This test depends on test__pickle_unpickle already having passed.

    pickle_filename = make_filename_in_artifacts_dir('test__unpickle_with_transform.pickle')
    delete_file_if_exists(pickle_filename)

    # First, create a pickle file.
    original_data = list(range(-100,101))
    transform = lambda data:[x**2.5 for x in data]
    transformed_data = transform(original_data)
    vorpy.pickle.pickle(data=original_data, pickle_filename=pickle_filename, log_out=sys.stdout)
    assert os.path.exists(pickle_filename)

    # Now attempt to unpickle and then pickle it.
    unpickled_data = vorpy.pickle.unpickle(pickle_filename=pickle_filename, transform_after_unpickle=transform, log_out=sys.stdout)

    # Ensure unpickle applied the transform correctly.
    assert transformed_data == unpickled_data

def test__unpickle_or_compute__having_to_compute ():
    expected_data = [x**2.5 for x in range(-100,101)]
    ran_computation = False
    def computation ():
        nonlocal ran_computation
        ran_computation = True
        return expected_data

    # Ensure the pickle file does not exist.
    pickle_filename = make_filename_in_artifacts_dir('test__unpickle_or_compute__having_to_compute.pickle')
    delete_file_if_exists(pickle_filename)

    actual_data = vorpy.pickle.__unpickle_or_compute(pickle_filename=pickle_filename, computation=computation, log_out=sys.stdout)

    # Ensure the computation ran and that the returned data matches expected.
    assert ran_computation
    assert actual_data == expected_data
    # Also assert that the pickle file (which didn't exist to begin with) still doesn't exist.
    assert not os.path.exists(pickle_filename)

def test__unpickle_or_compute__not_having_to_compute ():
    # This test depends on test__pickle_unpickle already having passed.

    expected_data = [x**2.5 for x in range(-100,101)]
    ran_computation = False
    def computation ():
        nonlocal ran_computation
        ran_computation = True
        return expected_data

    # First, create a pickle file.
    pickle_filename = make_filename_in_artifacts_dir('test__unpickle_or_compute__not_having_to_compute.pickle')
    vorpy.pickle.pickle(data=expected_data, pickle_filename=pickle_filename, log_out=sys.stdout)
    assert os.path.exists(pickle_filename)

    actual_data = vorpy.pickle.__unpickle_or_compute(pickle_filename=pickle_filename, computation=computation, log_out=sys.stdout)

    # Ensure the computation did not run and that the returned data matches expected.
    assert not ran_computation
    assert actual_data == expected_data

def test__unpickle_or_compute_and_try_to_pickle__not_having_to_compute ():
    # This test depends on test__pickle_unpickle already having passed.

    expected_data = [x**2.5 for x in range(-100,101)]
    ran_computation = False
    def computation ():
        nonlocal ran_computation
        ran_computation = True
        return expected_data

    # First, create a pickle file.
    pickle_filename = make_filename_in_artifacts_dir('test__unpickle_or_compute_and_try_to_pickle__not_having_to_compute.pickle')
    vorpy.pickle.pickle(data=expected_data, pickle_filename=pickle_filename, log_out=sys.stdout)
    assert os.path.exists(pickle_filename)

    actual_data = vorpy.pickle.unpickle_or_compute_and_try_to_pickle(pickle_filename=pickle_filename, computation=computation, log_out=sys.stdout)

    # Ensure the computation did not run and that the returned data matches expected.
    assert not ran_computation
    assert actual_data == expected_data

def test__unpickle_or_compute_and_try_to_pickle__having_to_compute ():
    expected_data = [x**2.5 for x in range(-100,101)]
    ran_computation = False
    def computation ():
        nonlocal ran_computation
        ran_computation = True
        return expected_data

    # Ensure the pickle file does not exist.
    pickle_filename = make_filename_in_artifacts_dir('test__unpickle_or_compute_and_pickle__having_to_compute.pickle')
    if os.path.exists(pickle_filename):
        os.remove(pickle_filename)

    actual_data = vorpy.pickle.unpickle_or_compute_and_try_to_pickle(pickle_filename=pickle_filename, computation=computation, log_out=sys.stdout)

    # Ensure the computation ran and that the returned data matches expected.
    assert ran_computation
    assert actual_data == expected_data
    # Also assert that the pickle file (which didn't exist to begin with) does exist now.
    assert os.path.exists(pickle_filename)

