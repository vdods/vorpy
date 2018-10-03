import numpy as np
import sympy as sp
import vorpy.linalg

def test_scalar_cross_product_tensor_symbolically ():
    scalar_cross_product_tensor = vorpy.linalg.scalar_cross_product_tensor(dtype=sp.Integer)
    u = np.array(sp.var('u_0,u_1'))
    v = np.array(sp.var('v_0,v_1'))
    # The awkward `[()]` notation on the left is to extract the scalar value from the 0-tensor result of np.cross
    # when it is given 2d vector arguments.  The awkward, nested np.dot expression on the right is to avoid the lack
    # of support for dtype=object in np.einsum.
    assert np.all(np.cross(u, v)[()] == np.dot(np.dot(scalar_cross_product_tensor, v), u))
    print('test_scalar_cross_product_tensor_symbolically passed.')

def test_scalar_cross_product_tensor_float ():
    scalar_cross_product_tensor = vorpy.linalg.scalar_cross_product_tensor(dtype=float)
    rng = np.random.RandomState(666)
    for i in range(100):
        u = rng.randn(2)
        for j in range(100):
            v = rng.randn(2)
            assert np.max(np.abs(np.cross(u,v)[()] - np.einsum('jk,j,k', scalar_cross_product_tensor, u, v))) < 1.0e-10
    print('test_scalar_cross_product_tensor_float passed.')

def test_cross_product_tensor_symbolically ():
    cross_product_tensor = vorpy.linalg.cross_product_tensor(dtype=sp.Integer)
    u = np.array(sp.var('u_0,u_1,u_2'))
    v = np.array(sp.var('v_0,v_1,v_2'))
    # The awkward, nested np.dot expression on the right is to avoid the lack of support for dtype=object in np.einsum.
    assert np.all(np.cross(u, v) == np.dot(np.dot(cross_product_tensor, v), u))
    print('test_cross_product_tensor_symbolically passed.')

def test_cross_product_tensor_float ():
    cross_product_tensor = vorpy.linalg.cross_product_tensor(dtype=float)
    rng = np.random.RandomState(666)
    for i in range(100):
        u = rng.randn(3)
        for j in range(100):
            v = rng.randn(3)
            assert np.max(np.abs(np.cross(u,v) - np.einsum('ijk,j,k', cross_product_tensor, u, v))) < 1.0e-10
    print('test_cross_product_tensor_float passed.')

if __name__ == '__main__':
    test_scalar_cross_product_tensor_symbolically()
    test_scalar_cross_product_tensor_float()
    test_cross_product_tensor_symbolically()
    test_cross_product_tensor_float()
