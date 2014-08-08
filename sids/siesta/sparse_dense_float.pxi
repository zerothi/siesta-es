#!python

# Create inline code for calculating phase
@cython.profile(False)
cdef inline float complex f_ph(float[:] k, float[:] xij) nogil:
     cdef float complex zz
     cdef float ph = k[0]*xij[0]+k[1]*xij[1]+k[2]*xij[2]
     # Cast numbers
     zz.real = cosf(ph)
     zz.imag = sinf(ph)
     return zz

@cython.profile(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def todense1_float(int is_gamma, int no_u, 
    np.ndarray[DINT_t,   ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_col not None,
    np.ndarray[DFLOAT_t, ndim=2, mode='c'] xij not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] k not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m not None):

    # views of inputs
    cdef int[:] n_colv = n_col
    cdef int[:] l_ptrv = l_ptr
    cdef int[:] l_colv = l_col
    cdef float[:,::1] xv = xij
    cdef float[:] mv = m

    cdef int io, jo, ind
    cdef float complex ik
    # Create views
    cdef float[:] kv = k

    cdef np.ndarray[DC_FLOAT_t, ndim=2] d = np.zeros([no_u,no_u], dtype=DC_FLOAT)
    cdef float complex[:,::1] dv = d

    if is_gamma == 0:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                ik = f_ph(kv,xv[ind,:])
                dv[io,jo].real = dv[io,jo].real + ik.real * mv[ind]
                dv[io,jo].imag = dv[io,jo].imag + ik.imag * mv[ind]
    else:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                dv[io,jo].real = dv[io,jo].real + mv[ind]
    return d

@cython.profile(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def todense2_float(int is_gamma, int no_u, 
    np.ndarray[DINT_t,   ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_col not None,
    np.ndarray[DFLOAT_t, ndim=2, mode='c'] xij not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] k not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m1 not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m2 not None):

    # views of inputs
    cdef int[:] n_colv = n_col
    cdef int[:] l_ptrv = l_ptr
    cdef int[:] l_colv = l_col
    cdef float[:,::1] xv = xij
    cdef float[:] m1v = m1
    cdef float[:] m2v = m2

    cdef int io, jo, ind
    cdef float complex ik
    # Create views
    cdef float[:] kv = k

    cdef np.ndarray[DC_FLOAT_t, ndim=2] d1 = np.zeros([no_u,no_u], dtype=DC_FLOAT)
    cdef np.ndarray[DC_FLOAT_t, ndim=2] d2 = np.zeros([no_u,no_u], dtype=DC_FLOAT)
    cdef float complex[:,::1] d1v = d1
    cdef float complex[:,::1] d2v = d2

    if is_gamma == 0:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                ik = f_ph(kv,xv[ind,:])
                d1v[io,jo].real = d1v[io,jo].real + ik.real * m1v[ind]
                d1v[io,jo].imag = d1v[io,jo].imag + ik.imag * m1v[ind]
                d2v[io,jo].real = d2v[io,jo].real + ik.real * m2v[ind]
                d2v[io,jo].imag = d2v[io,jo].imag + ik.imag * m2v[ind]
    else:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                d1v[io,jo].real = d1v[io,jo].real + m1v[ind]
                d2v[io,jo].real = d2v[io,jo].real + m2v[ind]
    return d1,d2

@cython.profile(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def todense1_float_off(int is_gamma, int no_u, 
    np.ndarray[DINT_t,   ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_col not None,
    np.ndarray[DFLOAT_t, ndim=2, mode='c'] off not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] k not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m not None):

    # views of inputs
    cdef int[:] n_colv = n_col
    cdef int[:] l_ptrv = l_ptr
    cdef int[:] l_colv = l_col
    cdef float[:] mv = m
    cdef float[:,::1] offv = off

    cdef int io, jo, ind
    cdef float complex ik
    # Create views
    cdef float[:] kv = k

    cdef np.ndarray[DC_FLOAT_t, ndim=2] d = np.zeros([no_u,no_u], dtype=DC_FLOAT)
    cdef float complex[:,::1] dv = d

    if is_gamma == 0:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                ik = f_ph(kv,offv[l_colv[ind] / no_u])
                dv[io,jo].real = dv[io,jo].real + ik.real * mv[ind]
                dv[io,jo].imag = dv[io,jo].imag + ik.imag * mv[ind]
    else:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                dv[io,jo].real = dv[io,jo].real + mv[ind]
    return d

@cython.profile(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def todense2_float_off(int is_gamma, int no_u, 
    np.ndarray[DINT_t,   ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_col not None,
    np.ndarray[DFLOAT_t, ndim=2, mode='c'] off not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] k not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m1 not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m2 not None):

    # views of inputs
    cdef int[:] n_colv = n_col
    cdef int[:] l_ptrv = l_ptr
    cdef int[:] l_colv = l_col
    cdef float[:,::1] offv = off
    cdef float[:] m1v = m1
    cdef float[:] m2v = m2

    cdef int io, jo, ind
    cdef float complex ik
    # Create views
    cdef float[:] kv = k

    cdef np.ndarray[DC_FLOAT_t, ndim=2] d1 = np.zeros([no_u,no_u], dtype=DC_FLOAT)
    cdef np.ndarray[DC_FLOAT_t, ndim=2] d2 = np.zeros([no_u,no_u], dtype=DC_FLOAT)
    cdef float complex[:,::1] d1v = d1
    cdef float complex[:,::1] d2v = d2

    if is_gamma == 0:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                ik = f_ph(kv,offv[l_colv[ind] / no_u])
                d1v[io,jo].real = d1v[io,jo].real + ik.real * m1v[ind]
                d1v[io,jo].imag = d1v[io,jo].imag + ik.imag * m1v[ind]
                d2v[io,jo].real = d2v[io,jo].real + ik.real * m2v[ind]
                d2v[io,jo].imag = d2v[io,jo].imag + ik.imag * m2v[ind]
    else:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                d1v[io,jo].real = d1v[io,jo].real + m1v[ind]
                d2v[io,jo].real = d2v[io,jo].real + m2v[ind]
    return d1,d2
