#!python
#cython: profile=False, cdivision=True, boundscheck=False, wraparound=False

include "../def_cython.pxi"
include "../def_linalg.pxi"

include "se_lopez_double.pxi"

def self_energy(double complex ZE,
    np.ndarray[DC_DOUBLE_t, ndim=2, mode='c'] H0 not None, \
    np.ndarray[DC_DOUBLE_t, ndim=2, mode='c'] S0 not None, \
    np.ndarray[DC_DOUBLE_t, ndim=2, mode='c'] H1 not None, \
    np.ndarray[DC_DOUBLE_t, ndim=2, mode='c'] S1 not None, \
    method = 'lopez', double eps = 1.e-15 ):

    cdef int no_u

    # Get size of the Hamiltonian
    no_u = H0.shape[1]

    if method == 'lopez':
       return se_lopez_dbl(no_u, ZE, H0, S0, H1, S1, eps)

    raise NotImplementedError('Method '+method+' is not implemented yet.')

def self_energy_ortho(double complex ZE,
    np.ndarray[DC_DOUBLE_t, ndim=2, mode='c'] H0 not None, \
    np.ndarray[DC_DOUBLE_t, ndim=2, mode='c'] H1 not None, \
    method = 'lopez', double eps = 1.e-15 ):

    cdef int no_u

    # Get size of the Hamiltonian
    no_u = np.len(H0[0,:])

    if method == 'lopez':
       return se_lopez_ortho_dbl(no_u, ZE, H0, H1, eps)

    raise NotImplementedError('Method '+method+' is not implemented yet.')
