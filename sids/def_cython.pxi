
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

#@cython.profile(False)
#cdef inline int nint(double d) nogil: return <int>lround(d)
@cython.profile(False)
cdef inline int nint(double d) nogil: return <int>round(d)

# Create max-min functions
@cython.profile(False)
cdef inline int imax(int a, int b) nogil: return a if a >= b else b
@cython.profile(False)
cdef inline int imin(int a, int b) nogil: return a if a <= b else b

@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
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
