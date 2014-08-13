#!python
#cython: profile=False, cdivision=True, boundscheck=False, wraparound=False

import scipy.sparse as spar

include "../def_cython.pxi"

@cython.boundscheck(False)
cdef inline int find_idx(int *col,int no,int find) nogil:
    cdef int i
    for i in xrange(no):
        if col[i] == find: return i

include "sparse_float.pxi"
include "sparse_double.pxi"
include "sparse_xij.pxi"

@cython.cdivision(True)
def sparse_uc(int no_u, 
    np.ndarray[DINT_t,   ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_col not None):

    # views of inputs
    cdef int[:] n_colv = n_col
    cdef int[:] l_ptrv = l_ptr
    cdef int[:] l_colv = l_col
    cdef int new_nnzs

    cdef int io, ind, indn

    cdef np.ndarray[DINT_t] n_col_n = np.empty([no_u], dtype=DINT) 
    cdef np.ndarray[DINT_t] l_ptr_n = np.empty([no_u+1], dtype=DINT) 
    cdef np.ndarray[DINT_t] l_col_n
    cdef int[:] n_colnv = n_col_n
    cdef int[:] l_ptrnv = l_ptr_n

    l_ptrnv[0] = 0
    for io in xrange(no_u):
        ind = l_ptrv[io]
        n_colnv[io] = len(np.unique(l_col[ind:ind+n_colv[io]]%no_u))
        if io > 0:
            l_ptrnv[io] = l_ptrnv[io-1] + n_colnv[io-1]

    # update last index (used for creating csr sparse format)
    l_ptrnv[no_u] = l_ptrnv[no_u-1] + n_colnv[no_u-1]

    # number of elements
    new_nnzs = l_ptrnv[no_u]
    l_col_n = np.empty([new_nnzs],dtype=DINT)

    for io in xrange(no_u):
        ind = l_ptrv[io]
        indn = l_ptrnv[io]

        # Create indices
        l_col_n[indn:indn+n_colnv[io]] = np.unique(l_col[ind:ind+n_colv[io]]%no_u)

    return l_ptr_n,l_col_n


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

# Wrapper to handle correct data
def tosparse(np.ndarray[DDOUBLE_t, ndim=1, mode='c'] k not None, \
    int no_u, \
    np.ndarray[DINT_t, ndim=1, mode='c'] n_col not None, \
    np.ndarray[DINT_t, ndim=1, mode='c'] l_ptr not None, \
    np.ndarray[DINT_t, ndim=1, mode='c'] l_col not None, \
    np.ndarray xij not None, \
    np.ndarray[DINT_t,    ndim=1, mode='c'] sp_ptr not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] sp_col not None,
    np.ndarray m1 not None, m2 = None):
    cdef np.ndarray[DFLOAT_t, ndim=1, mode='c'] kf = np.array(k,dtype=DFLOAT)
    cdef int is_gamma
    cdef double[:] kv = k

    # Decipher whether it is a Gamma point
    is_gamma = 0
    if kv[0] == 0. and kv[1] == 0. and kv[2] == 0.:
        is_gamma = 1

    # We now differ the data-types for the numpy arrays
    if m1.dtype == DFLOAT:
        if m2 is None:
            return tosparse1_float (is_gamma,no_u, n_col, l_ptr, l_col, xij, sp_ptr, sp_col, kf, m1)
        else:
            return tosparse2_float (is_gamma,no_u, n_col, l_ptr, l_col, xij, sp_ptr, sp_col, kf, m1, m2)
    else:
        if m2 is None:
            return tosparse1_double(is_gamma,no_u, n_col, l_ptr, l_col, xij, sp_ptr, sp_col, k , m1)
        else:
            return tosparse2_double(is_gamma,no_u, n_col, l_ptr, l_col, xij, sp_ptr, sp_col, k , m1, m2)


# Wrapper to handle correct data
def tosparse_off(np.ndarray[DDOUBLE_t, ndim=1, mode='c'] k not None, \
    int no_u, \
    np.ndarray[DINT_t, ndim=1, mode='c'] n_col not None, \
    np.ndarray[DINT_t, ndim=1, mode='c'] l_ptr not None, \
    np.ndarray[DINT_t, ndim=1, mode='c'] l_col not None, \
    np.ndarray off not None, \
    np.ndarray[DINT_t,    ndim=1, mode='c'] sp_ptr not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] sp_col not None,
    np.ndarray m1 not None, m2 = None):
    cdef np.ndarray[DFLOAT_t, ndim=1, mode='c'] kf = np.array(k,dtype=DFLOAT)
    cdef int is_gamma
    cdef double[:] kv = k

    # Decipher whether it is a Gamma point
    is_gamma = 0
    if kv[0] == 0. and kv[1] == 0. and kv[2] == 0.:
        is_gamma = 1

    # We now differ the data-types for the numpy arrays
    if m1.dtype == DFLOAT:
        if m2 is None:
            return tosparse1_float_off (is_gamma,no_u, n_col, l_ptr, l_col, off, sp_ptr, sp_col, kf, m1)
        else:
            return tosparse2_float_off (is_gamma,no_u, n_col, l_ptr, l_col, off, sp_ptr, sp_col, kf, m1, m2)
    else:
        if m2 is None:
            return tosparse1_double_off(is_gamma,no_u, n_col, l_ptr, l_col, off, sp_ptr, sp_col, k , m1)
        else:
            return tosparse2_double_off(is_gamma,no_u, n_col, l_ptr, l_col, off, sp_ptr, sp_col, k , m1, m2)
