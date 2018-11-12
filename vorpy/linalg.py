import numpy as np
import typing
import vorpy.tensor

def scalar_cross_product_tensor (*, dtype=float) -> np.ndarray:
    """
    Returns the 2-tensor which defines the scalar cross product in R^2, meaning that

        np.cross(a, b)[()] == np.einsum('jk,j,k', scalar_cross_product_tensor(dtype=dtype), a, b)

    where a and b have shape (2,) and dtype is compatible with the dtypes of a and b.  The
    `[()]` syntax on the return value of np.cross is because it returns a numpy.ndarray having
    shape () (i.e. a 0-tensor, aka a scalar), but the return value should be the scalar itself.

    Note that because the cross product is defined as a tensor that is meant to contract with
    indices, it can be applied to arguments having tensor order higher than 1 (i.e. the operands
    don't have to be vectors).

    In particular, the scalar cross product tensor is

        [  0 1 ]
        [ -1 0 ]

    and for reference, the scalar cross product between vectors (a,b) and (x,y) is a*y - b*x.
    """
    retval = np.zeros((2,2), dtype=dtype)
    retval[0,1] = dtype( 1)
    retval[1,0] = dtype(-1)
    return retval

def cross_product_tensor (*, dtype=float) -> np.ndarray:
    """
    Returns the 3-tensor which defines the cross product in R^3, meaning that

        np.cross(a, b) == np.einsum('ijk,j,k->i', cross_product_tensor(dtype=dtype), a, b)

    where a and b have shape (3,) and dtype is compatible with the dtypes of a and b.

    Note that because the cross product is defined as a tensor that is meant to contract with
    indices, it can be applied to arguments having tensor order higher than 1 (i.e. the operands
    don't have to be vectors).
    """
    retval = np.zeros((3,3,3), dtype=dtype)
    retval[2,0,1] = dtype( 1)
    retval[2,1,0] = dtype(-1)
    retval[1,0,2] = dtype(-1)
    retval[1,2,0] = dtype( 1)
    retval[0,1,2] = dtype( 1)
    retval[0,2,1] = dtype(-1)
    return retval
