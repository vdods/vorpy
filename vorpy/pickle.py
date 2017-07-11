"""
This module provides convenience functions for pickling/unpickling data to/from disk, as well as
higher-level functions which compute said data if the pickle file doesn't exist.

The idea is to be able to conveniently load expensively-computed data from cache if it has already
been cached, or failing that, computing it and caching it.

Example:

    import vorpy

    def expensive_computation ():
        data = <some very expensive computation>
        return data

    # If 'data.pickle' exists, it will be unpickled and returned.  If there is an error reading
    # or unpickling it, then expensive_computation is run, the results are pickled to 'data.pickle',
    # and the result returned.  If pickling to 'data.pickle' fails, the function still returns
    # the data.  Messages are printed to the optionally specifiable log_out object which must have
    # `write` method.
    data = vorpy.pickle.unpickle_or_compute_and_try_to_pickle(
        pickle_filename='data.pickle',
        computation=expensive_computation,
        log_out=sys.stdout
    )
"""

import dill
import os

def __log (out, message):
    if out is not None:
        out.write(message)

def pickle (*, data, pickle_filename, transform_before_pickle=None, log_out=None):
    """
    Pickles the specified data to the named pickle file.

    -   pickle_filename is the filename to attempt to pickle to.
    -   transform_before_unpickle is an optional transformation to be applied to the data before pickling.
    -   log_out is an optional write-able object which status messages will be written to.

    Returns the data that was pickled (which is the transformed data if the transform was specified).
    Error is indicated via exception.
    """
    log = lambda message:__log(log_out, message)
    log('attempting to open "{0}" for writing.\n'.format(pickle_filename))
    with open(pickle_filename, 'wb') as f:
        log('successfully opened "{0}" for writing.\n'.format(pickle_filename))
        log('attempting to pickle data.\n')
        if transform_before_pickle is not None:
            data = transform_before_pickle(data)
        dill.dump(data, f)
        log('successfully pickled data.\n')
    return data

def unpickle (*, pickle_filename, transform_after_unpickle=None, log_out=None):
    """
    Unpickles data from the given filename.

    -   pickle_filename is the filename to attempt to unpickle from.
    -   transform_after_unpickle is an optional transformation to be applied to the data after unpickling.
    -   log_out is an optional write-able object which status messages will be written to.

    Returns the unpickled data.  Error is indicated via exception.
    """
    log = lambda message:__log(log_out, message)
    log('attempting to open "{0}" for reading.\n'.format(pickle_filename))
    with open(pickle_filename, 'rb') as f:
        log('successfully opened "{0}" for reading.\n'.format(pickle_filename))
        log('attempting to unpickle data.\n')
        data = dill.load(f)
        if transform_after_unpickle is not None:
            data = transform_after_unpickle(data)
        log('successfully unpickled.\n')
        log('returning unpickled data.\n')
        return data

def __compute (*, computation, log_out=None):
    """
    This is just a wrapper function which provides log messages for running the computation.
    """
    log = lambda message:__log(log_out, message)
    log('running computation.\n')
    data = computation()
    log('successfully ran computation.\n')
    return data

def try_to_pickle (*, data, pickle_filename, transform_before_pickle=None, log_out=None):
    """
    Attempts to pickle the specified data to the named pickle file.

    -   pickle_filename is the filename to attempt to pickle to.
    -   transform_before_unpickle is an optional transformation to be applied to the data before pickling.
    -   log_out is an optional write-able object which status messages will be written to.

    Return True if and only if the pickling succeeded.
    """
    log = lambda message:__log(log_out, message)
    try:
        pickle(data=data, pickle_filename=pickle_filename, transform_before_pickle=transform_before_pickle, log_out=log_out)
        return True
    except (IOError, FileNotFoundError) as e:
        log('failed to open "{0}" for writing; error was "{1}".\n'.format(pickle_filename, e))
        return False
    except Exception as e:
        log('opened "{0}" for writing, but failed to pickle data; error was "{1}".\n'.format(pickle_filename, e))
        if os.path.exists(pickle_filename):
            log('deleting failed pickle file "{0}".\n'.format(pickle_filename))
            os.remove(pickle_filename)
        return False

def __compute_and_try_to_pickle (*, pickle_filename, computation, transform_before_pickle, log_out=None):
    log = lambda message:__log(log_out, message)
    data = __compute(computation=computation, log_out=log_out)
    try_to_pickle(data=data, pickle_filename=pickle_filename, transform_before_pickle=transform_before_pickle, log_out=log_out)
    log('returning computed data.\n')
    return data

def __unpickle_or_compute (*, pickle_filename, computation, transform_after_unpickle=None, log_out=None):
    """
    Attempts to unpickle data from the given filename, and if that fails, it will run the given computation
    to produce the desired data.  The idea here is that if there is an expensive computation, then it can
    be cached the first time it's computed, and then just re-loaded from cache later instead of being computed
    again.

    -   pickle_filename is the filename to attempt to unpickle from.
    -   computation must be a 0-parameter function which produces the data should the unpickling fail for whatever reason.
    -   transform_after_unpickle is an optional transformation to be applied to the data after unpickling.
    -   log_out is an optional write-able object which status messages will be written to.

    Returns the unpickled or computed data.
    """
    log = lambda message:__log(log_out, message)
    try:
        return unpickle(pickle_filename=pickle_filename, transform_after_unpickle=transform_after_unpickle, log_out=log_out)
    except (IOError, FileNotFoundError) as e:
        log('failed to open "{0}" for reading; error was "{1}".\n'.format(pickle_filename, e))
    except Exception as e:
        log('opened "{0}" for reading, but failed load pickle; error was "{1}".\n'.format(pickle_filename))
    return __compute(computation=computation, log_out=log_out)

def unpickle_or_compute_and_try_to_pickle (*, pickle_filename, computation, transform_after_unpickle=None, transform_before_pickle=None, log_out=None):
    """
    Attempts to unpickle data from the given filename, and if that fails, it will run the given computation
    to produce the desired data and pickles the resulting data in addition to returning it.  The idea here is
    that if there is an expensive computation, then it can be cached the first time it's computed, and then just
    re-loaded from cache later instead of being computed again.

    -   pickle_filename is the filename to attempt to unpickle from.
    -   computation must be a 0-parameter function which produces the data should the unpickling fail for whatever reason.
    -   transform_after_unpickle is an optional transformation to be applied to the data after unpickling.
    -   log_out is an optional write-able object which status messages will be written to.

    Returns the unpickled or computed data.
    """
    log = lambda message:__log(log_out, message)
    try:
        return unpickle(pickle_filename=pickle_filename, transform_after_unpickle=transform_after_unpickle, log_out=log_out)
    except (IOError, FileNotFoundError) as e:
        log('failed to open "{0}" for reading; error was "{1}".\n'.format(pickle_filename, e))
    except Exception as e:
        log('opened "{0}" for reading, but failed load pickle; error was "{1}".\n'.format(pickle_filename))
    return __compute_and_try_to_pickle(pickle_filename=pickle_filename, computation=computation, transform_before_pickle=transform_before_pickle, log_out=log_out)
