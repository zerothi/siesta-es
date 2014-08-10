#!python

# These routines helps conveting a SIESTA
# sparsity xij array to an appropriate super-cell index

@cython.profile(False)
@cython.boundscheck(False)
cdef inline int iaorb(int io,int na_u, int[:] lasto) nogil:
    cdef int i
    for i in xrange(na_u):
        if io < lasto[i]: return i
    return na_u

@cython.profile(False)
@cython.boundscheck(False)
cdef inline double xorcell(double x,double y, double z, double[:] rcell) nogil:
    return x*rcell[0] + x*rcell[1] + z*rcell[2]

@cython.profile(False)
@cython.boundscheck(False)
cdef inline double xorcellf(float x, float y, float z, double[:] rcell) nogil:
    return (<double>x)*rcell[0] + (<double>y)*rcell[1] + (<double>z)*rcell[2]

@cython.profile(False)
@cython.boundscheck(False)
cdef inline double i_mult_d(int i,double d) nogil: return (<double>i)*d
@cython.profile(False)
@cython.boundscheck(False)
cdef inline double i_mult_f(int i,float d) nogil: return (<float>i)*d

@cython.profile(False)
@cython.boundscheck(False)
def get_isupercells(np.ndarray[DINT_t] tm not None):
    cdef int i, x, y, z
    cdef np.ndarray[DINT_t, ndim=2] off = np.empty(((2*tm[0]+1)*(2*tm[1]+1)*(2*tm[2]+1),3),dtype=DINT)
    cdef int[:,::1] offv = off
    offv[0,0] = 0
    offv[0,1] = 0
    offv[0,2] = 0
    i = 0
    for z in xrange(-tm[2],tm[2]+1):
        for y in xrange(-tm[1],tm[1]+1):
            for x in xrange(-tm[0],tm[0]+1):
                if x == 0 and y == 0 and z == 0: continue
                i = i + 1
                offv[i,0] = x
                offv[i,1] = y
                offv[i,2] = z
    return off

@cython.profile(False)
@cython.boundscheck(False)
def get_supercells(np.ndarray[DDOUBLE_t,ndim=2] cell not None,
                   np.ndarray[DINT_t] tm not None):
    cdef int i, x, y, z
    cdef np.ndarray[DDOUBLE_t, ndim=2] off = np.empty(((2*tm[0]+1)*(2*tm[1]+1)*(2*tm[2]+1),3),dtype=DDOUBLE)
    cdef double[:,::1] offv = off
    cdef double[:,::1] cellv = cell
    offv[0,0] = 0.
    offv[0,1] = 0.
    offv[0,2] = 0.
    i = 0
    for z in xrange(-tm[2],tm[2]+1):
        for y in xrange(-tm[1],tm[1]+1):
            for x in xrange(-tm[0],tm[0]+1):
                if x == 0 and y == 0 and z == 0: continue
                i = i + 1
                offv[i,0] = i_mult_d(x,cellv[0,0])+i_mult_d(y,cellv[1,0])+i_mult_d(z,cellv[2,0])
                offv[i,1] = i_mult_d(x,cellv[0,1])+i_mult_d(y,cellv[1,1])+i_mult_d(z,cellv[2,1])
                offv[i,2] = i_mult_d(x,cellv[0,2])+i_mult_d(y,cellv[1,2])+i_mult_d(z,cellv[2,2])
    return off

@cython.profile(False)
@cython.boundscheck(False)
def get_supercellsf(np.ndarray[DDOUBLE_t,ndim=2] cell not None,
                   np.ndarray[DINT_t] tm not None):
    cdef int i, x, y, z
    cdef np.ndarray[DFLOAT_t, ndim=2] off = np.empty(((2*tm[0]+1)*(2*tm[1]+1)*(2*tm[2]+1),3),dtype=DFLOAT)
    cdef float[:,::1] offv = off
    cdef double[:,::1] cellv = cell
    offv[0,0] = 0.
    offv[0,1] = 0.
    offv[0,2] = 0.
    i = 0
    for z in xrange(-tm[2],tm[2]+1):
        for y in xrange(-tm[1],tm[1]+1):
            for x in xrange(-tm[0],tm[0]+1):
                if x == 0 and y == 0 and z == 0: continue
                i = i + 1
                offv[i,0] = <float>(i_mult_d(x,cellv[0,0])+i_mult_d(y,cellv[1,0])+i_mult_d(z,cellv[2,0]))
                offv[i,1] = <float>(i_mult_d(x,cellv[0,1])+i_mult_d(y,cellv[1,1])+i_mult_d(z,cellv[2,1]))
                offv[i,2] = <float>(i_mult_d(x,cellv[0,2])+i_mult_d(y,cellv[1,2])+i_mult_d(z,cellv[2,2]))
    return off

@cython.profile(False)
@cython.boundscheck(False)
def xij_correct(int na_u, 
           np.ndarray[DDOUBLE_t, ndim=2, mode='c'] xa not None,
           np.ndarray[DINT_t,    ndim=1, mode='c'] lasto not None,
           int no_u, 
           np.ndarray[DINT_t,    ndim=1, mode='c'] n_col not None,
           np.ndarray[DINT_t,    ndim=1, mode='c'] l_ptr not None,
           np.ndarray[DINT_t,    ndim=1, mode='c'] l_col not None,
           np.ndarray[DDOUBLE_t, ndim=2, mode='c'] xij not None):

    # views of inputs
    cdef double[:,::1] xav = xa
    cdef int[:] lastov = lasto
    cdef int[:] n_colv = n_col
    cdef int[:] l_ptrv = l_ptr
    cdef int[:] l_colv = l_col
    cdef double[:,::1] xijv = xij

    cdef int io, ind, ia, ja

    for io in xrange(no_u):
        ia = iaorb(io,na_u,lastov)
        for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
            ja = iaorb(l_colv[ind] % no_u,na_u,lastov)
            xijv[ind,0] = xijv[ind,0] - ( xav[ja,0] - xav[ia,0] )
            xijv[ind,1] = xijv[ind,1] - ( xav[ja,1] - xav[ia,1] )
            xijv[ind,2] = xijv[ind,2] - ( xav[ja,2] - xav[ia,2] )

@cython.profile(False)
@cython.boundscheck(False)
def xij_correctf(int na_u, 
           np.ndarray[DFLOAT_t, ndim=2, mode='c'] xa not None,
           np.ndarray[DINT_t,   ndim=1, mode='c'] lasto not None,
           int no_u, 
           np.ndarray[DINT_t,   ndim=1, mode='c'] n_col not None,
           np.ndarray[DINT_t,   ndim=1, mode='c'] l_ptr not None,
           np.ndarray[DINT_t,   ndim=1, mode='c'] l_col not None,
           np.ndarray[DFLOAT_t, ndim=2, mode='c'] xij not None):

    # views of inputs
    cdef float[:,::1] xav = xa
    cdef int[:] lastov = lasto
    cdef int[:] n_colv = n_col
    cdef int[:] l_ptrv = l_ptr
    cdef int[:] l_colv = l_col
    cdef float[:,::1] xijv = xij

    cdef int io, ind, ia, ja

    for io in xrange(no_u):
        ia = iaorb(io,na_u,lastov)
        for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
            ja = iaorb(l_colv[ind] % no_u,na_u,lastov)
            xijv[ind,0] = xijv[ind,0] - ( xav[ja,0] - xav[ia,0] )
            xijv[ind,1] = xijv[ind,1] - ( xav[ja,1] - xav[ia,1] )
            xijv[ind,2] = xijv[ind,2] - ( xav[ja,2] - xav[ia,2] )


@cython.profile(False)
@cython.boundscheck(False)
def xij_sc(np.ndarray[DDOUBLE_t, ndim=2, mode='c'] rcell not None,
           int nnzs, 
           np.ndarray[DDOUBLE_t, ndim=2, mode='c'] xij not None):

    # views of inputs
    cdef double[:] rcellx = rcell[0,:]
    cdef double[:] rcelly = rcell[1,:]
    cdef double[:] rcellz = rcell[2,:]
    cdef double[:,::1] xijv = xij

    cdef int ind
    cdef np.ndarray[DINT_t] tm = np.zeros((3,),dtype=DINT)
    cdef int[:] tmv = tm

    cdef int xc

    for ind in xrange(nnzs):
        xc = abs(nint(xorcell(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellx)))
        if xc > tmv[0]: tmv[0] = xc
        xc = abs(nint(xorcell(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcelly)))
        if xc > tmv[1]: tmv[1] = xc
        xc = abs(nint(xorcell(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellz)))
        if xc > tmv[2]: tmv[2] = xc

    return tm


@cython.profile(False)
@cython.boundscheck(False)
def xij_scf(np.ndarray[DDOUBLE_t, ndim=2, mode='c'] rcell not None,
           int nnzs,
           np.ndarray[DFLOAT_t, ndim=2, mode='c'] xij not None):

    # views of inputs
    cdef double[:] rcellx = rcell[0,:]
    cdef double[:] rcelly = rcell[1,:]
    cdef double[:] rcellz = rcell[2,:]
    cdef float[:,::1] xijv = xij

    cdef int ind
    cdef np.ndarray[DINT_t] tm = np.zeros((3,),dtype=DINT)
    cdef int[:] tmv = tm

    cdef int xc

    for ind in xrange(nnzs):
        xc = abs(nint(xorcellf(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellx)))
        if xc > tmv[0]: tmv[0] = xc
        xc = abs(nint(xorcellf(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcelly)))
        if xc > tmv[1]: tmv[1] = xc
        xc = abs(nint(xorcellf(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellz)))
        if xc > tmv[2]: tmv[2] = xc
    return tm

# Small inline functions to check transfer matrix
@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int tm_same(int[:] tm1, int[:] tm2) nogil:
    if tm1[0] != tm2[0]: return 0
    if tm1[1] != tm2[1]: return 0
    if tm1[2] != tm2[2]: return 0
    return 1

@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int offset_is(int[:] tm,int[:] tm2, int[:] ctm) nogil:
    cdef int i
    if ctm[2] > 0:
        i = 0
    elif ctm[1] > 0 and ctm[2] == 0:
        i = 0
    elif ctm[0] >= 0 and ctm[1] == 0 and ctm[2] == 0:
        if ctm[0] == 0: return 0
        i = 0
    else:
        i = 1
    i = i + (ctm[2] + tm[2]) * tm2[1] * tm2[0]
    i = i + (ctm[1] + tm[1]) * tm2[0]
    return i + ctm[0] + tm[0]

@cython.profile(False)
@cython.boundscheck(False)
def list_col_correct(np.ndarray[DDOUBLE_t, ndim=2, mode='c'] rcell not None,
                     int no_u, int nnzs,
                     np.ndarray[DINT_t,    ndim=1, mode='c'] l_col not None,
                     np.ndarray[DDOUBLE_t, ndim=2, mode='c'] xij not None,
                     np.ndarray[DINT_t,    ndim=1, mode='c'] tm not None,
                     np.ndarray[DINT_t,    ndim=2, mode='c'] off not None):

    # views of inputs
    cdef double[:] rcellx = rcell[0,:]
    cdef double[:] rcelly = rcell[1,:]
    cdef double[:] rcellz = rcell[2,:]
    cdef int[:] l_colv = l_col
    cdef double[:,::1] xijv = xij
    cdef int[:] tmv = tm
    cdef int[:] tm2v = tm * 2 + 1
    cdef int[:,::1] offv = off

    cdef int n_s = len(off[:,0])
    cdef int ind, si
    cdef int[:] ctm = np.empty((3,),dtype=DINT)

    for ind in xrange(nnzs):
        ctm[0] = nint(xorcell(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellx))
        ctm[1] = nint(xorcell(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcelly))
        ctm[2] = nint(xorcell(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellz))
        # We need to find the correct supercell index
        # AND correct l_col
        si = offset_is(tmv,tm2v,ctm)
        l_colv[ind] = si * no_u + l_colv[ind] % no_u

@cython.profile(False)
@cython.boundscheck(False)
def list_col_correctf(np.ndarray[DDOUBLE_t, ndim=2, mode='c'] rcell not None,
                     int no_u, int nnzs, 
                     np.ndarray[DINT_t,     ndim=1, mode='c'] l_col not None,
                     np.ndarray[DFLOAT_t,   ndim=2, mode='c'] xij not None,
                     np.ndarray[DINT_t,     ndim=1, mode='c'] tm not None,
                     np.ndarray[DINT_t,     ndim=2, mode='c'] off not None):

    # views of inputs
    cdef double[:] rcellx = rcell[0,:]
    cdef double[:] rcelly = rcell[1,:]
    cdef double[:] rcellz = rcell[2,:]
    cdef int[:] l_colv = l_col
    cdef float[:,::1] xijv = xij
    cdef int[:] tmv = tm
    cdef int[:] tm2v = tm * 2 + 1
    cdef int[:,::1] offv = off

    cdef int n_s = len(off[:,0])
    cdef int ind, si
    cdef int[:] ctm = np.empty((3,),dtype=DINT)

    for ind in xrange(nnzs):
        ctm[0] = nint(xorcellf(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellx))
        ctm[1] = nint(xorcellf(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcelly))
        ctm[2] = nint(xorcellf(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellz))
        # We need to find the correct supercell index
        # AND correct l_col
        si = offset_is(tmv,tm2v,ctm)
        l_colv[ind] = si * no_u + l_colv[ind] % no_u
