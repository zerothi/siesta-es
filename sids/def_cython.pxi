
# Allows compiling constructs
cimport cython

# Import numpy
import numpy as np
cimport numpy as np

# Math functions (ensures c-library math calls)
# We also ensure that they do not express any gil dependency
cdef extern from "stdlib.h" nogil:
    int abs(int)
cdef extern from "math.h" nogil:
    float lroundf(float)
cdef extern from "math.h" nogil:
    double sin(double)
cdef extern from "math.h" nogil:
    double cos(double)
cdef extern from "math.h" nogil:
    float sinf(float)
cdef extern from "math.h" nogil:
    float cosf(float)
cdef extern from "math.h" nogil:
    long int lround(double)
cdef extern from "math.h" nogil:
    double round(double)

cdef inline int nint(double d) nogil: return <int>round(d)

# Create max-min functions
cdef inline int imax(int a, int b) nogil: return a if a >= b else b
cdef inline int imin(int a, int b) nogil: return a if a <= b else b

cdef inline void rec_cell(double[:,::1] recell, double[:,::1] cell) nogil:
     cdef double tmp
     recell[0,0] = cell[1,1]*cell[2,2] - cell[1,2]*cell[2,1]
     recell[0,1] = cell[1,2]*cell[2,0] - cell[1,0]*cell[2,2]
     recell[0,2] = cell[1,0]*cell[2,1] - cell[1,1]*cell[2,0]
     recell[1,0] = cell[2,1]*cell[0,2] - cell[2,2]*cell[0,1]
     recell[1,1] = cell[2,2]*cell[0,0] - cell[2,0]*cell[0,2]
     recell[1,2] = cell[2,0]*cell[0,1] - cell[2,1]*cell[0,0]
     recell[2,0] = cell[0,1]*cell[1,2] - cell[0,2]*cell[1,1]
     recell[2,1] = cell[0,2]*cell[1,0] - cell[0,0]*cell[1,2]
     recell[2,2] = cell[0,0]*cell[1,1] - cell[0,1]*cell[1,0]
     tmp = 1./(cell[0,0]*recell[0,0]+cell[0,1]*recell[0,1]+cell[0,2]*recell[0,2])
     recell[0,0] = recell[0,0]*tmp
     recell[0,1] = recell[0,1]*tmp
     recell[0,2] = recell[0,2]*tmp
     tmp = 1./(cell[1,0]*recell[1,0]+cell[1,1]*recell[1,1]+cell[1,2]*recell[1,2])
     recell[1,0] = recell[1,0]*tmp
     recell[1,1] = recell[1,1]*tmp
     recell[1,2] = recell[1,2]*tmp
     tmp = 1./(cell[2,0]*recell[2,0]+cell[2,1]*recell[2,1]+cell[2,2]*recell[2,2])
     recell[2,0] = recell[2,0]*tmp
     recell[2,1] = recell[2,1]*tmp
     recell[2,2] = recell[2,2]*tmp

cdef inline double iMd(int i,double d) nogil: return (<double>i)*d
cdef inline double iMf(int i,float d) nogil: return (<float>i)*d

@cython.cdivision(True)
cdef inline int sfind(int no, int *a, int val) nogil:
    # my SFIND algorithm used in SIESTA
    cdef int idx, h, nom1 = no-1 # default index
    if no == 0: return -1

    # The two easiest cases, i.e. they are not in the array...
    if val < a[0]: return -1
    if val == a[0]: return 0
    if a[nom1] < val: return -1
    if val == a[nom1]: return nom1

    # An *advanced* search algorithm...
    
    # Search the sorted array for an entry
    # We know it *must* have one
    h = no / 2
    idx = h # Start in the middle
    # The integer correction (due to round of errors when 
    # calculating the new half...
    while h > 0:
        
        # integer division is faster. :)
        h = h / 2 + h % 2
                
        if val < a[idx]:
            # the value we search for is smaller than 
            # the current checked value, hence we step back
            # print 'stepping down',idx,h,no
            idx = imax(idx - h,0)
        elif a[idx] < val:
            # the value we search for is larger than 
            # the current checked value, hence we step forward
            # print 'stepping up',idx,h,no
            idx = imin(idx + h,nom1)
        else:
            # print 'found',idx
            # We know EXACTLY where we are...
            return idx
    return -1

cdef inline int uniqc(int no, int *a, int n, int* tmp) nogil:
    # my SFIND algorithm used in SIESTA
    cdef int i,j,nn,found
    if no == 0: return 0
    tmp[0] = a[0]
    nn = 1
    for i in xrange(no):
        found = 0
        for j in xrange(nn):
           if tmp[j] == a[i]: 
              found = 1
        if found == 0:
           nn = nn + 1
           tmp[nn] = a[i]
    return nn

# This is my first attempt in writing a compliant
# Cython code that takes a SIESTA sparse
# format and creates a matrix

# Create the two available data types
DINT = np.int32
DLONG = np.int64
DFLOAT = np.float32
DDOUBLE = np.float64
DC_FLOAT = np.complex64
DC_DOUBLE = np.complex128

ctypedef np.int32_t DINT_t
ctypedef np.int64_t DLONG_t
ctypedef np.float32_t DFLOAT_t
ctypedef np.float64_t DDOUBLE_t
ctypedef np.complex64_t DC_FLOAT_t
ctypedef np.complex128_t DC_DOUBLE_t
