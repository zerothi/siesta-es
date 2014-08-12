#!python

# Create inline code for calculating phase
@cython.profile(False)
cdef inline float complex f_ph(float[:] k, float x, float y, float z) nogil:
     cdef float complex zz
     cdef float ph = k[0]*x+k[1]*y+k[2]*z
     # Cast numbers
     zz.real = cosf(ph)
     zz.imag = sinf(ph)
     return zz

@cython.cdivision(True)
def tosparse1_float(int is_gamma, int no_u, 
    np.ndarray[DINT_t,    ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_col not None,
    np.ndarray[DFLOAT_t, ndim=2, mode='c'] xij not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] sp_ptr not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] sp_col not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] k not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m not None):

    # views of inputs
    cdef int[:] n_colv = n_col, l_ptrv = l_ptr, l_colv = l_col
    cdef int[:] sp_ptrv = sp_ptr, sp_colv = sp_col
    cdef float[:,::1] xv = xij
    cdef float[:] mv = m

    cdef int io, jo, ind, n
    cdef float complex ik
    # Create views
    cdef float[:] kv = k
    cdef int *col_ptr

    cdef np.ndarray[DC_FLOAT_t] d = np.zeros([sp_ptr[no_u]], dtype=DC_FLOAT)
    cdef float complex[:] dv = d

    if is_gamma == 0:
        for io in xrange(no_u):
            col_ptr = &sp_colv[sp_ptrv[io]]
            n = sp_ptrv[io+1] - sp_ptrv[io]
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                ik = f_ph(kv,xv[ind,0],xv[ind,1],xv[ind,2])
                jo = sp_ptrv[io] + find_idx(col_ptr,n,l_colv[ind] % no_u)
                dv[jo].real = dv[jo].real + ik.real * mv[ind]
                dv[jo].imag = dv[jo].imag + ik.imag * mv[ind]
    else:
        for io in xrange(no_u):
            col_ptr = &sp_colv[sp_ptrv[io]]
            n = sp_ptrv[io+1] - sp_ptrv[io]
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = sp_ptrv[io] + find_idx(col_ptr,n,l_colv[ind] % no_u)
                dv[jo].real = dv[jo].real + mv[ind]
    return spar.csr_matrix((d,sp_col,sp_ptr),shape=(no_u,no_u),dtype=DC_FLOAT)

@cython.cdivision(True)
def tosparse1_float_off(int is_gamma, int no_u, 
    np.ndarray[DINT_t,    ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_col not None,
    np.ndarray[DFLOAT_t, ndim=2, mode='c'] off not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] sp_ptr not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] sp_col not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] k not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m not None):

    # views of inputs
    cdef int[:] n_colv = n_col, l_ptrv = l_ptr, l_colv = l_col
    cdef int[:] sp_ptrv = sp_ptr, sp_colv = sp_col
    cdef float[:,::1] offv = off
    cdef float[:] mv = m

    cdef int io, jo, ind, si, n
    cdef float complex ik
    # Create views
    cdef float[:] kv = k
    cdef int *col_ptr

    cdef np.ndarray[DC_FLOAT_t] d = np.zeros([sp_ptr[no_u]], dtype=DC_FLOAT)
    cdef float complex[:] dv = d

    if is_gamma == 0:
        for io in xrange(no_u):
            col_ptr = &sp_colv[sp_ptrv[io]]
            n = sp_ptrv[io+1] - sp_ptrv[io]
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                si = l_colv[ind] / no_u
                ik = f_ph(kv,offv[si,0],offv[si,1],offv[si,2])
                jo = sp_ptrv[io] + find_idx(col_ptr,n,l_colv[ind] % no_u)
                dv[jo].real = dv[jo].real + ik.real * mv[ind]
                dv[jo].imag = dv[jo].imag + ik.imag * mv[ind]
    else:
        for io in xrange(no_u):
            col_ptr = &sp_colv[sp_ptrv[io]]
            n = sp_ptrv[io+1] - sp_ptrv[io]
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = sp_ptrv[io] + find_idx(col_ptr,n,l_colv[ind] % no_u)
                dv[jo].real = dv[jo].real + mv[ind]
    return spar.csr_matrix((d,sp_col,sp_ptr),shape=(no_u,no_u),dtype=DC_FLOAT)


@cython.cdivision(True)
def tosparse2_float(int is_gamma, int no_u, 
    np.ndarray[DINT_t,    ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_col not None,
    np.ndarray[DFLOAT_t, ndim=2, mode='c'] xij not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] sp_ptr not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] sp_col not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] k not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m1 not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m2 not None):

    # views of inputs
    cdef int[:] n_colv = n_col, l_ptrv = l_ptr, l_colv = l_col
    cdef int[:] sp_ptrv = sp_ptr, sp_colv = sp_col
    cdef float[:,::1] xv = xij
    cdef float[:] m1v = m1, m2v = m2

    cdef int io, jo, ind, n
    cdef float complex ik
    # Create views
    cdef float[:] kv = k
    cdef int *col_ptr

    cdef np.ndarray[DC_FLOAT_t] d1 = np.zeros([sp_ptr[no_u]], dtype=DC_FLOAT)
    cdef np.ndarray[DC_FLOAT_t] d2 = np.zeros([sp_ptr[no_u]], dtype=DC_FLOAT)
    cdef float complex[:] d1v = d1, d2v = d2

    if is_gamma == 0:
        for io in xrange(no_u):
            col_ptr = &sp_colv[sp_ptrv[io]]
            n = sp_ptrv[io+1] - sp_ptrv[io]
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                ik = f_ph(kv,xv[ind,0],xv[ind,1],xv[ind,2])
                jo = sp_ptrv[io] + find_idx(col_ptr,n,l_colv[ind] % no_u)
                d1v[jo].real = d1v[jo].real + ik.real * m1v[ind]
                d1v[jo].imag = d1v[jo].imag + ik.imag * m1v[ind]
                d2v[jo].real = d2v[jo].real + ik.real * m2v[ind]
                d2v[jo].imag = d2v[jo].imag + ik.imag * m2v[ind]
    else:
        for io in xrange(no_u):
            col_ptr = &sp_colv[sp_ptrv[io]]
            n = sp_ptrv[io+1] - sp_ptrv[io]
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = sp_ptrv[io] + find_idx(col_ptr,n,l_colv[ind] % no_u)
                d1v[jo].real = d1v[jo].real + m1v[ind]
                d2v[jo].real = d2v[jo].real + m2v[ind]

    return (spar.csr_matrix((d1,sp_col,sp_ptr),shape=(no_u,no_u),dtype=DC_FLOAT),
            spar.csr_matrix((d2,sp_col,sp_ptr),shape=(no_u,no_u),dtype=DC_FLOAT))


@cython.cdivision(True)
def tosparse2_float_off(int is_gamma, int no_u, 
    np.ndarray[DINT_t,    ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] l_col not None,
    np.ndarray[DFLOAT_t, ndim=2, mode='c'] off not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] sp_ptr not None,
    np.ndarray[DINT_t,    ndim=1, mode='c'] sp_col not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] k not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m1 not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m2 not None):

    # views of inputs
    cdef int[:] n_colv = n_col, l_ptrv = l_ptr, l_colv = l_col
    cdef int[:] sp_ptrv = sp_ptr, sp_colv = sp_col
    cdef float[:,::1] offv = off
    cdef float[:] m1v = m1, m2v = m2

    cdef int io, jo, ind, si, n
    cdef float complex ik
    # Create views
    cdef float[:] kv = k
    cdef int *col_ptr

    cdef np.ndarray[DC_FLOAT_t] d1 = np.zeros([sp_ptr[no_u]], dtype=DC_FLOAT)
    cdef np.ndarray[DC_FLOAT_t] d2 = np.zeros([sp_ptr[no_u]], dtype=DC_FLOAT)
    cdef float complex[:] d1v = d1, d2v = d2

    if is_gamma == 0:
        for io in xrange(no_u):
            col_ptr = &sp_colv[sp_ptrv[io]]
            n = sp_ptrv[io+1] - sp_ptrv[io]
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                si = l_colv[ind] / no_u
                ik = f_ph(kv,offv[si,0],offv[si,1],offv[si,2])
                jo = sp_ptrv[io] + find_idx(col_ptr,n,l_colv[ind] % no_u)
                d1v[jo].real = d1v[jo].real + ik.real * m1v[ind]
                d1v[jo].imag = d1v[jo].imag + ik.imag * m1v[ind]
                d2v[jo].real = d2v[jo].real + ik.real * m2v[ind]
                d2v[jo].imag = d2v[jo].imag + ik.imag * m2v[ind]
    else:
        for io in xrange(no_u):
            col_ptr = &sp_colv[sp_ptrv[io]]
            n = sp_ptrv[io+1] - sp_ptrv[io]
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = sp_ptrv[io] + find_idx(col_ptr,n,l_colv[ind] % no_u)
                d1v[jo].real = d1v[jo].real + m1v[ind]
                d2v[jo].real = d2v[jo].real + m2v[ind]

    return (spar.csr_matrix((d1,sp_col,sp_ptr),shape=(no_u,no_u),dtype=DC_FLOAT),
            spar.csr_matrix((d2,sp_col,sp_ptr),shape=(no_u,no_u),dtype=DC_FLOAT))

@cython.cdivision(True)
def todense1_float(int is_gamma, int no_u, 
    np.ndarray[DINT_t,   ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_col not None,
    np.ndarray[DFLOAT_t, ndim=2, mode='c'] xij not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] k not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m not None):

    # views of inputs
    cdef int[:] n_colv = n_col, l_ptrv = l_ptr, l_colv = l_col
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
                ik = f_ph(kv,xv[ind,0],xv[ind,1],xv[ind,2])
                dv[io,jo].real = dv[io,jo].real + ik.real * mv[ind]
                dv[io,jo].imag = dv[io,jo].imag + ik.imag * mv[ind]
    else:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                dv[io,jo].real = dv[io,jo].real + mv[ind]
    return d

@cython.cdivision(True)
def todense2_float(int is_gamma, int no_u, 
    np.ndarray[DINT_t,   ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_col not None,
    np.ndarray[DFLOAT_t, ndim=2, mode='c'] xij not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] k not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m1 not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m2 not None):

    # views of inputs
    cdef int[:] n_colv = n_col, l_ptrv = l_ptr, l_colv = l_col
    cdef float[:,::1] xv = xij
    cdef float[:] m1v = m1, m2v = m2

    cdef int io, jo, ind
    cdef float complex ik
    # Create views
    cdef float[:] kv = k

    cdef np.ndarray[DC_FLOAT_t, ndim=2] d1 = np.zeros([no_u,no_u], dtype=DC_FLOAT)
    cdef np.ndarray[DC_FLOAT_t, ndim=2] d2 = np.zeros([no_u,no_u], dtype=DC_FLOAT)
    cdef float complex[:,::1] d1v = d1, d2v = d2

    if is_gamma == 0:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                ik = f_ph(kv,xv[ind,0],xv[ind,1],xv[ind,2])
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

@cython.cdivision(True)
def todense1_float_off(int is_gamma, int no_u, 
    np.ndarray[DINT_t,   ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_col not None,
    np.ndarray[DFLOAT_t, ndim=2, mode='c'] off not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] k not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m not None):

    # views of inputs
    cdef int[:] n_colv = n_col, l_ptrv = l_ptr, l_colv = l_col
    cdef float[:] mv = m
    cdef float[:,::1] offv = off

    cdef int io, jo, ind, si
    cdef float complex ik
    # Create views
    cdef float[:] kv = k

    cdef np.ndarray[DC_FLOAT_t, ndim=2] d = np.zeros([no_u,no_u], dtype=DC_FLOAT)
    cdef float complex[:,::1] dv = d

    if is_gamma == 0:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                si = l_colv[ind] / no_u
                ik = f_ph(kv,offv[si,0],offv[si,1],offv[si,2])
                dv[io,jo].real = dv[io,jo].real + ik.real * mv[ind]
                dv[io,jo].imag = dv[io,jo].imag + ik.imag * mv[ind]
    else:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                dv[io,jo].real = dv[io,jo].real + mv[ind]
    return d

@cython.cdivision(True)
def todense2_float_off(int is_gamma, int no_u, 
    np.ndarray[DINT_t,   ndim=1, mode='c'] n_col not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_ptr not None,
    np.ndarray[DINT_t,   ndim=1, mode='c'] l_col not None,
    np.ndarray[DFLOAT_t, ndim=2, mode='c'] off not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] k not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m1 not None,
    np.ndarray[DFLOAT_t, ndim=1, mode='c'] m2 not None):

    # views of inputs
    cdef int[:] n_colv = n_col, l_ptrv = l_ptr, l_colv = l_col
    cdef float[:,::1] offv = off
    cdef float[:] m1v = m1, m2v = m2

    cdef int io, jo, ind, si
    cdef float complex ik
    # Create views
    cdef float[:] kv = k

    cdef np.ndarray[DC_FLOAT_t, ndim=2] d1 = np.zeros([no_u,no_u], dtype=DC_FLOAT)
    cdef np.ndarray[DC_FLOAT_t, ndim=2] d2 = np.zeros([no_u,no_u], dtype=DC_FLOAT)
    cdef float complex[:,::1] d1v = d1, d2v = d2

    if is_gamma == 0:
        for io in xrange(no_u):
            for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
                jo = l_colv[ind] % no_u
                si = l_colv[ind] / no_u
                ik = f_ph(kv,offv[si,0],offv[si,1],offv[si,2])
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
