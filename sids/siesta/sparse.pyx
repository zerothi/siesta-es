#!python
#cython: profile=False, cdivision=True, boundscheck=False, wraparound=False

include "../def_cython.pxi"

include "sparse_dense_float.pxi"
include "sparse_dense_double.pxi"
include "sparse_xij.pxi"

# Wrapper to handle correct data
def todense(np.ndarray[DDOUBLE_t, ndim=1, mode='c'] k not None, \
    int no_u, \
    np.ndarray[DINT_t, ndim=1, mode='c'] n_col not None, \
    np.ndarray[DINT_t, ndim=1, mode='c'] l_ptr not None, \
    np.ndarray[DINT_t, ndim=1, mode='c'] l_col not None, \
    np.ndarray xij not None, \
    np.ndarray m1 not None, m2 = None):
    cdef np.ndarray[DFLOAT_t, ndim=1, mode='c'] kf = np.array(k,dtype=DFLOAT)
    cdef int is_gamma

    # Decipher whether it is a Gamma point
    is_gamma = 0
    if k[0] == 0. and k[1] == 0. and k[2] == 0.:
        is_gamma = 1

    # We now differ the data-types for the numpy arrays
    if m1.dtype == DFLOAT:
        if m2 is None:
            return todense1_float (is_gamma,no_u, n_col, l_ptr, l_col, xij, kf, m1)
        else:
            return todense2_float (is_gamma,no_u, n_col, l_ptr, l_col, xij, kf, m1, m2)
    else:
        if m2 is None:
            return todense1_double(is_gamma,no_u, n_col, l_ptr, l_col, xij, k , m1)
        else:
            return todense2_double(is_gamma,no_u, n_col, l_ptr, l_col, xij, k , m1, m2)

# Wrapper to handle correct data
def todense_off(np.ndarray[DDOUBLE_t, ndim=1, mode='c'] k not None, \
    int no_u, \
    np.ndarray[DINT_t, ndim=1, mode='c'] n_col not None, \
    np.ndarray[DINT_t, ndim=1, mode='c'] l_ptr not None, \
    np.ndarray[DINT_t, ndim=1, mode='c'] l_col not None, \
    np.ndarray off not None, \
    np.ndarray m1 not None, m2 = None):
    cdef np.ndarray[DFLOAT_t, ndim=1, mode='c'] kf = np.array(k,dtype=DFLOAT)
    cdef int is_gamma

    # Decipher whether it is a Gamma point
    is_gamma = 0
    if k[0] == 0. and k[1] == 0. and k[2] == 0.:
        is_gamma = 1

    # We now differ the data-types for the numpy arrays
    if m1.dtype == DFLOAT:
        if m2 is None:
            return todense1_float_off (is_gamma,no_u, n_col, l_ptr, l_col, off, kf, m1)
        else:
            return todense2_float_off (is_gamma,no_u, n_col, l_ptr, l_col, off, kf, m1, m2)
    else:
        if m2 is None:
            return todense1_double_off(is_gamma,no_u, n_col, l_ptr, l_col, off, k , m1)
        else:
            return todense2_double_off(is_gamma,no_u, n_col, l_ptr, l_col, off, k , m1, m2)
