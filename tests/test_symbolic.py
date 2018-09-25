import itertools
import numpy as np
import os
import shutil
import sympy
import time
import vorpy.symbolic

TEST_ARTIFACTS_DIR = 'test_artifacts/symbolic'

def make_filename_in_artifacts_dir (filename):
    os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)
    return os.path.join(TEST_ARTIFACTS_DIR, filename)

def benchmark_stuff (y_, f_, phi_, A_):
    rng = np.random.RandomState(666)

    tick_time = time.time()
    def tick ():
        nonlocal tick_time
        current_time = time.time()
        retval = current_time - tick_time
        tick_time = current_time
        return retval

    tick()
    for x_ in np.linspace(0.0, 10.0, 123456):
        assert y_(x_) == x_**(1/7)
    y_benchmark = tick()

    tick()
    for v_ in map(np.array, itertools.product(*map(np.linspace, [-1.0]*3, [1.0]*3, [23]*3))):
        assert f_(v_) == sum(v_[i]**2 for i in range(3))
    f_benchmark = tick()

    tick()
    for v_ in map(np.array, itertools.product(*map(np.linspace, [-1.0]*3, [1.0]*3, [23]*3))):
        norm_v_ = np.linalg.norm(v_)
        # Avoid divide by zero or near zero.
        if norm_v_ < 1.0e-10:
            continue
        max_abs_error = np.max(np.abs(phi_(v_) - np.array([v_[i] / norm_v_ for i in range(3)])))
        assert max_abs_error < 1.0e-14, 'v_ = {0}, max_abs_error = {1}'.format(v_, max_abs_error)
    phi_benchmark = tick()

    tick()
    for _ in range(10000):
        M_ = rng.randn(6,5)
        max_abs_error = np.max(np.abs(A_(M_) - M_.T.dot(M_)))
        assert max_abs_error < 1.0e-14, 'M_ = {0}, max_abs_error = {1}'.format(M_, max_abs_error)
    A_benchmark = tick()

    return y_benchmark,f_benchmark,phi_benchmark,A_benchmark

def test_lambdified ():
    tick_time = time.time()
    def tick ():
        nonlocal tick_time
        current_time = time.time()
        retval = current_time - tick_time
        tick_time = current_time
        return retval

    print('test_lambdified()')

    x = vorpy.symbolic.variable('x')
    y = x**sympy.Rational(1,7)
    tick()
    y_ = vorpy.symbolic.lambdified(y, x, replacement_d={'array':'np.array', 'ndarray':'np.ndarray', 'dtype=object':'dtype=float'}, verbose=True)
    y_lambdified_time = tick()

    v = vorpy.symbolic.tensor('v', (3,))
    f = np.sum(np.square(v))
    tick()
    f_ = vorpy.symbolic.lambdified(f, v, replacement_d={'array':'np.array', 'ndarray':'np.ndarray', 'dtype=object':'dtype=float'}, verbose=True)
    f_lambdified_time = tick()

    phi = v / sympy.sqrt(np.sum(np.square(v)))
    tick()
    phi_ = vorpy.symbolic.lambdified(phi, v, replacement_d={'array':'np.array', 'ndarray':'np.ndarray', 'dtype=object':'dtype=float', 'sqrt':'np.sqrt'}, verbose=True)
    phi_lambdified_time = tick()

    M = vorpy.symbolic.tensor('M', (6,5))
    A = M.T.dot(M)
    tick()
    A_ = vorpy.symbolic.lambdified(A, M, replacement_d={'array':'np.array', 'ndarray':'np.ndarray', 'dtype=object':'dtype=float'}, verbose=True)
    A_lambdified_time = tick()

    y_benchmark,f_benchmark,phi_benchmark,A_benchmark = benchmark_stuff(y_, f_, phi_, A_)

    print('y, f, phi, A lambdified time:', y_lambdified_time, f_lambdified_time, phi_lambdified_time, A_lambdified_time)
    print('y, f, phi, A benchmarks     :', y_benchmark, f_benchmark, phi_benchmark, A_benchmark)

    print('unit test passed.')

def test_lambdified_cached ():
    tick_time = time.time()
    def tick ():
        nonlocal tick_time
        current_time = time.time()
        retval = current_time - tick_time
        tick_time = current_time
        return retval

    def run (cache_dirname=None, use_numba=False):
        def y_creator ():
            x = vorpy.symbolic.variable('x')
            y = x**sympy.Rational(1,7)
            import_v = ['import numpy as np']
            decorator_v = []
            if use_numba:
                import_v.append('import numba')
                decorator_v.append('@numba.jit("float64(float64)", cache=True, nopython=True)')
                # decorator_v.append('@numba.jit("float64(float64)", cache=True)')
            return y,x,{'array':'np.array', 'ndarray':'np.ndarray', 'dtype=object':'dtype=float'},'x',import_v,decorator_v

        tick()
        y_ = vorpy.symbolic.cached_lambdified('y', function_creator=y_creator, cache_dirname=cache_dirname, verbose=True)
        y_lambdified_time = tick()

        v_dim = 3

        def f_creator ():
            v = vorpy.symbolic.tensor('v', (v_dim,))
            f = np.sum(np.square(v))
            import_v = ['import numpy as np']
            decorator_v = []
            if use_numba:
                import_v.append('import numba')
                decorator_v.append('@numba.jit("float64(float64[:])", cache=True, nopython=True)')
                # decorator_v.append('@numba.jit("float64(float64[:])", cache=True)')
            return f,v,{'array':'np.array', 'ndarray':'np.ndarray', 'dtype=object':'dtype=float'},'v',import_v,decorator_v

        tick()
        f_ = vorpy.symbolic.cached_lambdified('f', function_creator=f_creator, cache_dirname=cache_dirname, verbose=True)
        f_lambdified_time = tick()

        def phi_creator ():
            v = vorpy.symbolic.tensor('v', (v_dim,))
            phi = v / sympy.sqrt(np.sum(np.square(v)))
            import_v = ['import numpy as np']
            decorator_v = []
            if use_numba:
                import_v.append('import numba')
                # decorator_v.append('@numba.jit("float64[:](float64[:])", cache=True, nopython=True)')
                decorator_v.append('@numba.jit("float64[:](float64[:])", cache=True)')
                # decorator_v.append('@numba.jit(locals=dict(retval=numba.float64[:], v=numba.float64[:]), cache=True, nopython=True)')
            return phi,v,{'array':'np.array', 'ndarray':'np.ndarray', 'dtype=object':'dtype=float', 'sqrt':'np.sqrt'},'v',import_v,decorator_v

        tick()
        phi_ = vorpy.symbolic.cached_lambdified('phi', function_creator=phi_creator, cache_dirname=cache_dirname, verbose=True)
        phi_lambdified_time = tick()

        M_shape = (6,5)

        def A_creator ():
            M = vorpy.symbolic.tensor('M', M_shape)
            A = M.T.dot(M)
            import_v = ['import numpy as np']
            decorator_v = []
            if use_numba:
                import_v.append('import numba')
                # decorator_v.append('@numba.jit("float64[:,:](float64[:,:])", cache=True, nopython=True)')
                decorator_v.append('@numba.jit("float64[:,:](float64[:,:])", cache=True)')
                # decorator_v.append('@numba.jit(locals=dict(retval=numba.float64[:,:], M=numba.float64[:,:]), cache=True, nopython=True)')
            return A,M,{'array':'np.array', 'ndarray':'np.ndarray', 'dtype=object':'dtype=float'},'M',import_v,decorator_v

        tick()
        A_ = vorpy.symbolic.cached_lambdified('A', function_creator=A_creator, cache_dirname=cache_dirname, verbose=True)
        A_lambdified_time = tick()

        y_benchmark,f_benchmark,phi_benchmark,A_benchmark = benchmark_stuff(y_, f_, phi_, A_)
        print('y, f, phi, A lambdified time:', y_lambdified_time, f_lambdified_time, phi_lambdified_time, A_lambdified_time)
        print('y, f, phi, A benchmarks     :', y_benchmark, f_benchmark, phi_benchmark, A_benchmark)

    def run_uncached_and_cached (cache_dirname=None, use_numba=False, cached_run_iteration_count=1):
        # Ensure the cache is deleted, if it existed already.
        if os.path.exists(cache_dirname):
            print('deleting cache dir "{0}".'.format(cache_dirname))
            shutil.rmtree(cache_dirname)

        # Run the first time where the functions must be generated and cached.
        print('running unit test with no cached lambdas.')
        start_time = time.time()
        run(cache_dirname=cache_dirname, use_numba=use_numba)
        print('unit test with no cached lambdas took {0} seconds.'.format(time.time() - start_time))
        print('')

        for iteration_index in range(cached_run_iteration_count):
            assert os.path.exists(cache_dirname)

            # Run the second time where the functions can be loaded from the cache.
            print('running unit test with all cached lambdas (iteration {0} out of {1})'.format(iteration_index+1, cached_run_iteration_count))
            start_time = time.time()
            run(cache_dirname=cache_dirname, use_numba=use_numba)
            print('unit test with all cached lambdas took {0} seconds.'.format(time.time() - start_time))
            print('')

    print('test_lambdified_cached()')
    print('')

    cache_dirname = 'test_lambdified_cached'
    run_uncached_and_cached(cache_dirname=cache_dirname, use_numba=False, cached_run_iteration_count=3)
    if os.path.exists(cache_dirname):
        print('deleting cache dir "{0}".'.format(cache_dirname))
        shutil.rmtree(cache_dirname)
    print('')

    cache_dirname = 'test_lambdified_cached_numba'
    run_uncached_and_cached(cache_dirname=cache_dirname, use_numba=True, cached_run_iteration_count=3)
    if os.path.exists(cache_dirname):
        print('deleting cache dir "{0}".'.format(cache_dirname))
        shutil.rmtree(cache_dirname)
    print('')

    print('unit test passed.')

def test_homogeneous_polynomial ():
    x,y,z = X = np.array((sympy.var('x'), sympy.var('y'), sympy.var('z')))

    degree = 0
    h0, C0 = vorpy.symbolic.homogeneous_polynomial('a', degree, X)
    print(f'homogeneous polynomial of degree {degree} in variables {X} is {h0}, and it has coefficients {C0}')
    assert np.shape(C0) == (1,)
    assert np.all(h0 == C0[0])

    degree = 1
    h1, C1 = vorpy.symbolic.homogeneous_polynomial('a', degree, X)
    print(f'homogeneous polynomial of degree {degree} in variables {X} is {h1}, and it has coefficients {C1}')
    assert np.shape(C1) == (3,)
    assert np.all(h1 == C1[0]*x + C1[1]*y + C1[2]*z)

    degree = 2
    h2, C2 = vorpy.symbolic.homogeneous_polynomial('a', degree, X)
    print(f'homogeneous polynomial of degree {degree} in variables {X} is {h2}, and it has coefficients {C2}')
    assert np.shape(C2) == (6,)
    assert np.all(h2 == C2[0]*x**2 + C2[1]*x*y + C2[2]*x*z + C2[3]*y**2 + C2[4]*y*z + C2[5]*z**2)

    print('test_homogeneous_polynomial passed')

def test_polynomial ():
    x,y,z = X = np.array((sympy.var('x'), sympy.var('y'), sympy.var('z')))

    assert vorpy.symbolic.homogeneous_polynomial('a', 0, X)[0] == vorpy.symbolic.polynomial('a', range(1), X)[0]
    assert vorpy.symbolic.homogeneous_polynomial('a', 1, X)[0] == vorpy.symbolic.polynomial('a', range(2), X)[0] - vorpy.symbolic.polynomial('a', range(1), X)[0]
    assert vorpy.symbolic.homogeneous_polynomial('a', 2, X)[0] == vorpy.symbolic.polynomial('a', range(3), X)[0] - vorpy.symbolic.polynomial('a', range(2), X)[0]
    assert vorpy.symbolic.homogeneous_polynomial('a', 3, X)[0] == vorpy.symbolic.polynomial('a', range(4), X)[0] - vorpy.symbolic.polynomial('a', range(3), X)[0]

    h0, C0 = vorpy.symbolic.homogeneous_polynomial('a', 0, X)
    h2, C2 = vorpy.symbolic.homogeneous_polynomial('a', 2, X)
    h5, C5 = vorpy.symbolic.homogeneous_polynomial('a', 5, X)

    p, C = vorpy.symbolic.polynomial('a', [0, 2, 5], X)
    assert p == h0 + h2 + h5
    assert C == C0 + C2 + C5

    print('test_polynomial passed')

if __name__ == '__main__':
    test_homogeneous_polynomial()
    test_polynomial()

