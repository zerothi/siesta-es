#!python
#cython: profile=False, cdivision=True, boundscheck=False, wraparound=False

# We here create an algorithm for calculating the
# self-energies using pure C code

# Wrapper to handle correct data
cdef se_lopez_dbl(int no_u, double complex ZE, \
                      np.ndarray[DC_DOUBLE_t, ndim=2, mode='c'] H0, \
                      np.ndarray[DC_DOUBLE_t, ndim=2, mode='c'] S0, \
                      np.ndarray[DC_DOUBLE_t, ndim=2, mode='c'] H1, \
                      np.ndarray[DC_DOUBLE_t, ndim=2, mode='c'] S1, \
                      double eps):

    cdef double complex[::1,:] alpha = np.empty([no_u,no_u], dtype=DC_DOUBLE).T
    cdef double complex[::1,:] beta  = np.empty([no_u,no_u], dtype=DC_DOUBLE).T
    cdef double complex[::1,:] gb    = np.empty([no_u,no_u], dtype=DC_DOUBLE).T

    cdef int io, jo

    for io in xrange(no_u):
        for jo in xrange(no_u):
            gb[jo,io]    = ZE * S0[jo,io] - H0[jo,io]
            alpha[jo,io] = H1[jo,io] - ZE * S1[jo,io]
            beta[jo,io]  = H1[io,jo].real - H1[io,jo].imag - \
                ZE * (S1[io,jo].real - S1[io,jo].imag)

    return lopez_dbl(no_u,eps, alpha, beta, gb)

# Wrapper to handle correct data
cdef se_lopez_ortho_dbl(int no_u, double complex ZE, \
                            np.ndarray[DC_DOUBLE_t, ndim=2, mode='c'] H0, \
                            np.ndarray[DC_DOUBLE_t, ndim=2, mode='c'] H1, \
                            double eps):

    cdef double complex[::1,:] alpha = np.empty([no_u,no_u], dtype=DC_DOUBLE).T
    cdef double complex[::1,:] beta  = np.empty([no_u,no_u], dtype=DC_DOUBLE).T
    cdef double complex[::1,:] gb    = np.empty([no_u,no_u], dtype=DC_DOUBLE).T

    cdef int io, jo

    for io in xrange(no_u):
        for jo in xrange(io):
            gb[jo,io]    = - H0[io,jo]
            alpha[jo,io] = H1[io,jo]
            beta[jo,io]  = H1[jo,io].real - H1[jo,io].imag
        gb[io,io]    = ZE - H0[io,io]
        alpha[io,io] = H1[io,io] - ZE
        beta[io,io]  = H1[io,io].real - H1[io,io].imag - ZE
        for jo in xrange(io+1,no_u):
            gb[jo,io]    = - H0[io,jo]
            alpha[jo,io] = H1[io,jo]
            beta[jo,io]  = H1[jo,io].real - H1[jo,io].imag
            
    return lopez_dbl(no_u,eps, alpha, beta, gb)

cdef lopez_dbl(int no_u, double eps, 
               double complex[::1,:] alpha, double complex[::1,:] beta,
               double complex[::1,:] gb):

    cdef int no2 = no_u * 2
    cdef double complex[::1,:] rh  = np.empty([no2,no_u], dtype=DC_DOUBLE).T
    cdef double complex[::1,:] rh1 = np.empty([no2,no_u], dtype=DC_DOUBLE).T
    cdef double complex[::1,:] w   = np.empty([no_u,no_u], dtype=DC_DOUBLE).T

    cdef np.ndarray[DC_DOUBLE_t, ndim=2] SEN = np.empty([no_u,no_u], dtype=DC_DOUBLE)
    cdef double complex[::1,:] SE = SEN.T
    cdef double complex z1  =  1. +1j*0.
    cdef double complex zm1 = -1. +1j*0.
    cdef double complex z0  =  0. +1j*0.

    cdef int io, jo , it
    cdef int[:] pvt = np.empty([no_u], dtype=DINT)
    cdef double ro

    # Copy over initial G
    SE[:,:] = gb[:,:]

    ro = eps + 1.
    it = 0
    while ro > eps:

        it = it + 1
        
        for jo in xrange(no_u):
            for io in xrange(no_u):
                rh[io,jo]      = alpha[io,jo]
                rh[io,jo+no_u] = beta [io,jo]
                w [io,jo]      = gb   [io,jo]

        # Solve system of equations g_nn x = [alpha beta]
        c_zgesv(w[:,:],pvt[:],rh[:,:])

        # The bulk Green's function g_nn
        c_zgemm('N','N',zm1,beta[:,:],rh[:,:no_u],z1,gb[:,:])
        #gb[:,:] -= np.dot(beta[:,:],rh[:,0:no_u])

        c_zgemm('N','N',zm1,alpha[:,:],rh[:,no_u:],z0,w[:,:])
        #w[:,:] = np.dot(alpha[:,:],rh[:,no_u:])

        for jo in xrange(no_u):
            for io in xrange(no_u):
                SE[io,jo] = SE[io,jo] + w[io,jo]
                gb[io,jo] = gb[io,jo] + w[io,jo]

        ro = np.amax(np.abs(w))

        c_zgemm('N','N',z1,alpha[:,:],rh[:,:no_u],z0,w[:,:])
        alpha[:,:] = w[:,:]
        c_zgemm('N','N',z1,beta[:,:],rh[:,no_u:],z0,w[:,:])
        beta[:,:] = w[:,:]
        #alpha[:,:] = np.dot(alpha[:,:],rh[:,:no_u])
        #beta[:,:]  = np.dot(beta[:,:],rh[:,no_u:])

    return it, SEN
