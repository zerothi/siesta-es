#!python

# Create inline code for calculating phase
@cython.profile(False)
cdef inline double complex d_ph(double[:] k,double x,double y,double z) nogil:
     cdef double complex zz
     cdef double ph = k[0]*x+k[1]*y+k[2]*z
     # Cast numbers
     zz.real = cos(ph)
     zz.imag = sin(ph)
     return zz

@cython.profile(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def todense1_double(int is_gamma, int no_u, 
    np.ndarray[DINT_t,    ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_col not None,
    np.ndarray[DDOUBLE_t, ndim=2, mode='c'] xij not None,
    np.ndarray[DDOUBLE_t, ndim=1, mode='c'] k not None,
    np.ndarray[DDOUBLE_t, ndim=1, mode='c'] m not None):

    # views of inputs
    cdef int[:] n_colv = n_col
    cdef int[:] l_ptrv = l_ptr
    cdef int[:] l_colv = l_col
    cdef double[:,::1] xv = xij
    cdef double[:] mv = m

    cdef int io, jo, ind
    cdef double complex ik
    # Create views
    cdef double[:] kv = k

    cdef np.ndarray[DC_DOUBLE_t, ndim=2] d = np.zeros([no_u,no_u], dtype=DC_DOUBLE)
    cdef double complex[:,::1] dv = d

    if is_gamma == 0:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                ik = d_ph(kv,xv[ind,0],xv[ind,1],xv[ind,2])
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
def todense2_double(int is_gamma, int no_u, 
    np.ndarray[DINT_t,    ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_col not None,
    np.ndarray[DDOUBLE_t, ndim=2, mode='c'] xij not None,
    np.ndarray[DDOUBLE_t, ndim=1, mode='c'] k not None,
    np.ndarray[DDOUBLE_t, ndim=1, mode='c'] m1 not None,
    np.ndarray[DDOUBLE_t, ndim=1, mode='c'] m2 not None):

    # views of inputs
    cdef int[:] n_colv = n_col
    cdef int[:] l_ptrv = l_ptr
    cdef int[:] l_colv = l_col
    cdef double[:,::1] xv = xij
    cdef double[:] m1v = m1
    cdef double[:] m2v = m2

    cdef int io, jo, ind
    cdef double complex ik
    cdef double[:] kv = k

    cdef np.ndarray[DC_DOUBLE_t, ndim=2] d1 = np.zeros([no_u,no_u], dtype=DC_DOUBLE)
    cdef np.ndarray[DC_DOUBLE_t, ndim=2] d2 = np.zeros([no_u,no_u], dtype=DC_DOUBLE)
    cdef double complex[:,::1] d1v = d1
    cdef double complex[:,::1] d2v = d2

    if is_gamma == 0:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                ik = d_ph(kv,xv[ind,0],xv[ind,1],xv[ind,2])
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
def todense1_double_off(int is_gamma, int no_u, 
    np.ndarray[DINT_t,    ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_col not None,
    np.ndarray[DDOUBLE_t, ndim=2, mode='c'] off not None,
    np.ndarray[DDOUBLE_t, ndim=1, mode='c'] k not None,
    np.ndarray[DDOUBLE_t, ndim=1, mode='c'] m not None):

    # views of inputs
    cdef int[:] n_colv = n_col
    cdef int[:] l_ptrv = l_ptr
    cdef int[:] l_colv = l_col
    cdef double[:,::1] offv = off
    cdef double[:] mv = m

    cdef int io, jo, ind, si
    cdef double complex ik
    # Create views
    cdef double[:] kv = k

    cdef np.ndarray[DC_DOUBLE_t, ndim=2] d = np.zeros([no_u,no_u], dtype=DC_DOUBLE)
    cdef double complex[:,::1] dv = d

    if is_gamma == 0:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                si = l_colv[ind] / no_u
                ik = d_ph(kv,offv[si,0],offv[si,1],offv[si,2])
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
def todense2_double_off(int is_gamma, int no_u, 
    np.ndarray[DINT_t,    ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_col not None,
    np.ndarray[DDOUBLE_t, ndim=2, mode='c'] off not None,
    np.ndarray[DDOUBLE_t, ndim=1, mode='c'] k not None,
    np.ndarray[DDOUBLE_t, ndim=1, mode='c'] m1 not None,
    np.ndarray[DDOUBLE_t, ndim=1, mode='c'] m2 not None):

    # views of inputs
    cdef int[:] n_colv = n_col
    cdef int[:] l_ptrv = l_ptr
    cdef int[:] l_colv = l_col
    cdef double[:,::1] offv = off
    cdef double[:] m1v = m1
    cdef double[:] m2v = m2

    cdef int io, jo, ind, si
    cdef double complex ik
    cdef double[:] kv = k

    cdef np.ndarray[DC_DOUBLE_t, ndim=2] d1 = np.zeros([no_u,no_u], dtype=DC_DOUBLE)
    cdef np.ndarray[DC_DOUBLE_t, ndim=2] d2 = np.zeros([no_u,no_u], dtype=DC_DOUBLE)
    cdef double complex[:,::1] d1v = d1
    cdef double complex[:,::1] d2v = d2

    if is_gamma == 0:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                si = l_colv[ind] / no_u
                ik = d_ph(kv,offv[si,0],offv[si,1],offv[si,2])
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
