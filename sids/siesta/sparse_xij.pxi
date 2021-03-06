#!python

# These routines helps conveting a SIESTA
# sparsity xij array to an appropriate super-cell index

@cython.profile(False)
cdef inline int iaorb(int io,int na_u, int[:] lasto) nogil:
    cdef int i
    for i in xrange(na_u): 
        if io < lasto[i]: return i
    return na_u

@cython.profile(False)
cdef inline double xorcell(double x,double y, double z, double[:] rcell) nogil:
    return x * rcell[0] + y * rcell[1] + z * rcell[2]

@cython.profile(False)
cdef inline double xorcellf(float x, float y, float z, double[:] rcell) nogil:
    return (<double>x) * rcell[0] + (<double>y) * rcell[1] + (<double>z) * rcell[2]

@cython.profile(False)
cdef inline int n_offset(int[:] tmv) nogil:
    # Count the number of offsets used
    return (2*tmv[0]+1)*(2*tmv[1]+1)*(2*tmv[2]+1)

@cython.cdivision(True)
cdef inline void set_offset(int[:] tmv, int idx, int[:] o_tm) nogil:
    cdef int i
    # Create the transfer matrix position based on tmv
    o_tm[0] = idx % (2*tmv[0]+1)
    i = (idx - o_tm[0]) / (2*tmv[0]+1)
    o_tm[1] = i % (2*tmv[1]+1)
    o_tm[2] = (i - o_tm[1]) / (2*tmv[1]+1)
    o_tm[0] = o_tm[0] - tmv[0]
    o_tm[1] = o_tm[1] - tmv[1]
    o_tm[2] = o_tm[2] - tmv[2]

cdef inline int get_offset_index(int[:] tm,int[:] tm2, int[:] ctm) nogil:
    cdef int i
    i = (ctm[2]+tm[2])*tm2[1] + ctm[1] + tm[1]
    return i * tm2[0] + ctm[0] + tm[0]

# Small inline functions to check transfer matrix
cdef inline int tm_same(int[:] tm1, int[:] tm2) nogil:
    if tm1[0] != tm2[0]: return 0
    if tm1[1] != tm2[1]: return 0
    if tm1[2] != tm2[2]: return 0
    return 1

def get_isupercells(np.ndarray[DINT_t] tm not None):
    cdef int[:] tmv = tm
    cdef int i, n_s = n_offset(tmv)
    cdef np.ndarray[DINT_t, ndim=2] off = np.empty([n_s,3],dtype=DINT)
    cdef int[:,::1] offv = off

    for i in xrange(n_s):
        set_offset(tmv,i,offv[i,:])
    return off

def get_supercells(np.ndarray[DDOUBLE_t,ndim=2] cell not None,
                   np.ndarray[DINT_t] tm not None):
    cdef int[:] tmv = tm
    cdef int n_s = n_offset(tmv)
    cdef int i
    cdef np.ndarray[DDOUBLE_t, ndim=2] off = np.empty((n_s,3),dtype=DDOUBLE)
    cdef double[:,::1] offv = off
    cdef double[:,::1] cv = cell
    cdef int[:] o_tm = np.empty((3,),dtype=DINT)
    cdef int *x = &o_tm[0]
    cdef int *y = &o_tm[1]
    cdef int *z = &o_tm[2]

    for i in xrange(n_s):
        set_offset(tmv,i,o_tm)
        offv[i,0] = iMd(x[0],cv[0,0])+iMd(y[0],cv[1,0])+iMd(z[0],cv[2,0])
        offv[i,1] = iMd(x[0],cv[0,1])+iMd(y[0],cv[1,1])+iMd(z[0],cv[2,1])
        offv[i,2] = iMd(x[0],cv[0,2])+iMd(y[0],cv[1,2])+iMd(z[0],cv[2,2])

    return off

def xij_correct(int na_u, 
           np.ndarray xa not None,
           np.ndarray[DINT_t,    ndim=1, mode='c'] lasto not None,
           int no_u, 
           np.ndarray[DINT_t,    ndim=1, mode='c'] n_col not None,
           np.ndarray[DINT_t,    ndim=1, mode='c'] l_ptr not None,
           np.ndarray[DINT_t,    ndim=1, mode='c'] l_col not None,
           np.ndarray xij not None):
    if xij.dtype == DFLOAT:
        return xij_correctf(na_u,xa,lasto,no_u,n_col,l_ptr,l_col,xij)
    return xij_correctd(na_u,xa,lasto,no_u,n_col,l_ptr,l_col,xij)

cdef xij_correctd(int na_u, 
           np.ndarray[DDOUBLE_t, ndim=2, mode='c'] xa,
           np.ndarray[DINT_t,    ndim=1, mode='c'] lasto,
           int no_u, 
           np.ndarray[DINT_t,    ndim=1, mode='c'] n_col,
           np.ndarray[DINT_t,    ndim=1, mode='c'] l_ptr,
           np.ndarray[DINT_t,    ndim=1, mode='c'] l_col,
           np.ndarray[DDOUBLE_t, ndim=2, mode='c'] xij):

    # views of inputs
    cdef double[:,::1] xav = xa
    cdef int[:] lastov = lasto, n_colv = n_col, l_ptrv = l_ptr, l_colv = l_col
    cdef double[:,::1] xijv = xij

    cdef int io, ind, ia, ja

    for io in xrange(no_u):
        ia = iaorb(io,na_u,lastov)
        for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
            ja = iaorb(l_colv[ind] % no_u,na_u,lastov)
            xijv[ind,0] = xijv[ind,0] - ( xav[ja,0] - xav[ia,0] )
            xijv[ind,1] = xijv[ind,1] - ( xav[ja,1] - xav[ia,1] )
            xijv[ind,2] = xijv[ind,2] - ( xav[ja,2] - xav[ia,2] )


cdef xij_correctf(int na_u, 
           np.ndarray[DFLOAT_t, ndim=2, mode='c'] xa,
           np.ndarray[DINT_t,   ndim=1, mode='c'] lasto,
           int no_u, 
           np.ndarray[DINT_t,   ndim=1, mode='c'] n_col,
           np.ndarray[DINT_t,   ndim=1, mode='c'] l_ptr,
           np.ndarray[DINT_t,   ndim=1, mode='c'] l_col,
           np.ndarray[DFLOAT_t, ndim=2, mode='c'] xij):

    # views of inputs
    cdef float[:,::1] xav = xa
    cdef int[:] lastov = lasto, n_colv = n_col, l_ptrv = l_ptr, l_colv = l_col
    cdef float[:,::1] xijv = xij

    cdef int io, ind, ia, ja

    for io in xrange(no_u):
        ia = iaorb(io,na_u,lastov)
        for ind in xrange(l_ptrv[io],l_ptrv[io]+n_colv[io]):
            ja = iaorb(l_colv[ind] % no_u,na_u,lastov)
            xijv[ind,0] = xijv[ind,0] - ( xav[ja,0] - xav[ia,0] )
            xijv[ind,1] = xijv[ind,1] - ( xav[ja,1] - xav[ia,1] )
            xijv[ind,2] = xijv[ind,2] - ( xav[ja,2] - xav[ia,2] )

def xij_sc(np.ndarray[DDOUBLE_t, ndim=2, mode='c'] rcell not None,
           int nnzs, 
           np.ndarray xij not None):
    if xij.dtype == DFLOAT:
        return xij_scf(rcell,nnzs,xij)
    return xij_scd(rcell,nnzs,xij)


cdef xij_scd(np.ndarray[DDOUBLE_t, ndim=2, mode='c'] rcell,
           int nnzs, 
           np.ndarray[DDOUBLE_t, ndim=2, mode='c'] xij):

    # views of inputs
    cdef double[:] rcellx = rcell[:,0], rcelly = rcell[:,1], rcellz = rcell[:,2]
    cdef double[:,::1] xijv = xij

    cdef int ind, xc
    cdef np.ndarray[DINT_t] tm = np.zeros([3],dtype=DINT)
    cdef int[:] tmv = tm

    for ind in xrange(nnzs):
        xc = abs(nint(xorcell(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellx)))
        if xc > tmv[0]: tmv[0] = xc
        xc = abs(nint(xorcell(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcelly)))
        if xc > tmv[1]: tmv[1] = xc
        xc = abs(nint(xorcell(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellz)))
        if xc > tmv[2]: tmv[2] = xc

    return tm

cdef xij_scf(np.ndarray[DDOUBLE_t, ndim=2, mode='c'] rcell,
           int nnzs, 
           np.ndarray[DFLOAT_t, ndim=2, mode='c'] xij):

    # views of inputs
    cdef double[:] rcellx = rcell[:,0], rcelly = rcell[:,1], rcellz = rcell[:,2]
    cdef float[:,::1] xijv = xij

    cdef int ind, xc
    cdef np.ndarray[DINT_t] tm = np.zeros([3],dtype=DINT)
    cdef int[:] tmv = tm

    for ind in xrange(nnzs):
        xc = abs(nint(xorcellf(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellx)))
        if xc > tmv[0]: tmv[0] = xc
        xc = abs(nint(xorcellf(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcelly)))
        if xc > tmv[1]: tmv[1] = xc
        xc = abs(nint(xorcellf(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellz)))
        if xc > tmv[2]: tmv[2] = xc
    
    return tm


def list_col_correct(np.ndarray[DDOUBLE_t, ndim=2, mode='c'] rcell not None,
                     int no_u, int nnzs,
                     np.ndarray[DINT_t,    ndim=1, mode='c'] l_col not None,
                     np.ndarray xij not None,
                     np.ndarray[DINT_t,    ndim=1, mode='c'] tm not None):
    if xij.dtype == DFLOAT:
        return list_col_correctf(rcell,no_u,nnzs,l_col,xij,tm)
    return list_col_correctd(rcell,no_u,nnzs,l_col,xij,tm)


cdef list_col_correctd(np.ndarray[DDOUBLE_t, ndim=2, mode='c'] rcell,
                     int no_u, int nnzs,
                     np.ndarray[DINT_t,    ndim=1, mode='c'] l_col,
                     np.ndarray[DDOUBLE_t, ndim=2, mode='c'] xij,
                     np.ndarray[DINT_t,    ndim=1, mode='c'] tm):

    # views of inputs
    cdef double[:] rcellx = rcell[:,0], rcelly = rcell[:,1], rcellz = rcell[:,2]
    cdef int[:] l_colv = l_col
    cdef double[:,::1] xijv = xij
    cdef int[:] tmv = tm

    cdef int ind, si, jo
    cdef int[:] ctm = np.empty([3],dtype=DINT)
    cdef int[:] tm2 = np.empty([3],dtype=DINT)
    tm2[0] = tmv[0]*2+1
    tm2[1] = tmv[1]*2+1
    tm2[2] = tmv[2]*2+1

    for ind in xrange(nnzs):
        jo = l_colv[ind] % no_u
        ctm[0] = nint(xorcell(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellx))
        ctm[1] = nint(xorcell(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcelly))
        ctm[2] = nint(xorcell(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellz))
        # We need to find the correct supercell index
        # AND correct l_col
        si = get_offset_index(tmv,tm2,ctm)
        l_colv[ind] = si * no_u + jo

cdef list_col_correctf(np.ndarray[DDOUBLE_t, ndim=2, mode='c'] rcell,
                     int no_u, int nnzs,
                     np.ndarray[DINT_t,    ndim=1, mode='c'] l_col,
                     np.ndarray[DFLOAT_t,  ndim=2, mode='c'] xij,
                     np.ndarray[DINT_t,    ndim=1, mode='c'] tm):

    # views of inputs
    cdef double[:] rcellx = rcell[:,0], rcelly = rcell[:,1], rcellz = rcell[:,2]
    cdef int[:] l_colv = l_col
    cdef float[:,::1] xijv = xij
    cdef int[:] tmv = tm

    cdef int ind, si, jo
    cdef int[:] ctm = np.empty([3],dtype=DINT)
    cdef int[:] tm2 = np.empty([3],dtype=DINT)
    tm2[0] = tmv[0]*2+1
    tm2[1] = tmv[1]*2+1
    tm2[2] = tmv[2]*2+1

    for ind in xrange(nnzs):
        jo = l_colv[ind] % no_u
        ctm[0] = nint(xorcellf(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellx))
        ctm[1] = nint(xorcellf(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcelly))
        ctm[2] = nint(xorcellf(xijv[ind,0],xijv[ind,1],xijv[ind,2],rcellz))
        # We need to find the correct supercell index
        # AND correct l_col
        si = get_offset_index(tmv,tm2,ctm)
        l_colv[ind] = si * no_u + jo

