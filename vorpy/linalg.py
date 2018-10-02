import numpy as np

def cross_product_tensor (*, dtype=float):
    """
    Returns the tensor which defines the cross product in R^3, meaning that

        np.cross(a, b) == np.einsum('ijk,j,k->i', cross_product_tensor(dtype=dtype), a, b)

    a and b have shape (3,) and dtype is compatible with the dtypes of a and b.  Note that
    because the cross product is defined as a tensor that is meant to contract with indices,
    it can be applied to arguments having tensor order higher than 1 (i.e. the operands
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
