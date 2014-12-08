
cdef extern from "f2pyptr.h": 
    void *f2py_pointer(object) except NULL

##########################
# Imported BLAS routines #
##########################
import scipy.linalg.blas
 
ctypedef int sgemm_t(char *transa, char *transb,
	int *m, int *n, int *k,
	float *alpha, float *a, int *lda,
	float *b, int *ldb, float *beta,
	float *c, int *ldc)

ctypedef int dgemm_t(char *transa, char *transb,
	int *m, int *n, int *k,
	double *alpha, double *a, int *lda,
	double *b, int *ldb, double *beta,
	double *c, int *ldc)

ctypedef int cgemm_t(char *transa, char *transb,
	int *m, int *n, int *k,
	float complex *alpha, float complex *a, int *lda,
	float complex *b, int *ldb, float complex *beta,
	float complex *c, int *ldc)

ctypedef int zgemm_t(char *transa, char *transb,
	int *m, int *n, int *k,
	double complex *alpha, double complex *a, int *lda,
	double complex *b, int *ldb, double complex *beta,
	double complex *c, int *ldc)

# Since Scipy >= 0.12.0
cdef sgemm_t *sgemm = <sgemm_t*>f2py_pointer(scipy.linalg.blas.sgemm._cpointer)
cdef dgemm_t *dgemm = <dgemm_t*>f2py_pointer(scipy.linalg.blas.dgemm._cpointer)
cdef cgemm_t *cgemm = <cgemm_t*>f2py_pointer(scipy.linalg.blas.cgemm._cpointer)
cdef zgemm_t *zgemm = <zgemm_t*>f2py_pointer(scipy.linalg.blas.zgemm._cpointer)

cdef gemm_opts(char ta, char tb, int A0, int A1, int B0, int B1, int C0, int C1):
     cdef int m, n, k

     # Default sizes
     if ta == 'N':
        m, k = (A0,A1)
     else:
        m, k = (A1,A0)
     if tb == 'N':
        n = B1
     else:
        n = B0
     return m,n,k,A0,B0,C0

cdef c_sgemm(char ta, char tb,
     float alpha, float[::1,:] A, float[::1,:] B, float beta, float[::1,:] C):
     
     cdef int m, n, k, lda, ldb, ldc

     m, n, k, lda, ldb, ldc = gemm_opts(ta,tb,
     	A.shape[0],A.shape[1],B.shape[0],B.shape[1],C.shape[0],C.shape[1])

     sgemm(&ta, &tb, &m, &n, &k, &alpha, &A[0,0], &lda, &B[0,0], &ldb, &beta, &C[0,0], &ldc)

cdef c_dgemm(char ta, char tb,
     double alpha, double[::1,:] A, double[::1,:] B, 
     double beta, double[::1,:] C):
     
     cdef int m, n, k, lda, ldb, ldc

     m, n, k, lda, ldb, ldc = gemm_opts(ta,tb,
     	A.shape[0],A.shape[1],B.shape[0],B.shape[1],C.shape[0],C.shape[1])

     dgemm(&ta, &tb, &m, &n, &k, &alpha, &A[0,0], &lda, &B[0,0], &ldb, &beta, &C[0,0], &ldc)

cdef c_cgemm(char ta, char tb,
     float complex alpha, float complex[::1,:] A, float complex[::1,:] B, 
     float complex beta, float complex[::1,:] C):
     
     cdef int m, n, k, lda, ldb, ldc

     m, n, k, lda, ldb, ldc = gemm_opts(ta,tb,
     	A.shape[0],A.shape[1],B.shape[0],B.shape[1],C.shape[0],C.shape[1])

     cgemm(&ta, &tb, &m, &n, &k, &alpha, &A[0,0], &lda, &B[0,0], &ldb, &beta, &C[0,0], &ldc)

cdef c_zgemm(char ta, char tb,
     double complex alpha, double complex[::1,:] A, double complex[::1,:] B, 
     double complex beta, double complex[::1,:] C):
     
     cdef int m, n, k, lda, ldb, ldc

     m, n, k, lda, ldb, ldc = gemm_opts(ta,tb,
     	A.shape[0],A.shape[1],B.shape[0],B.shape[1],C.shape[0],C.shape[1])

     zgemm(&ta, &tb, &m, &n, &k, &alpha, &A[0,0], &lda, &B[0,0], &ldb, &beta, &C[0,0], &ldc)


############################
# Imported LAPACK routines #
############################
import scipy.linalg.lapack

ctypedef int sgesv_t(int *N, int *NRSH, float *A, int *lda, 
	 int *piv, float *B, int *ldb, int *info)
ctypedef int dgesv_t(int *N, int *NRSH, double *A, int *lda,
	 int *piv, double *B, int *ldb, int *info)
ctypedef int cgesv_t(int *N, int *NRSH, float complex *A, int *lda, 
	 int *piv, float complex *B, int *ldb, int *info)
ctypedef int zgesv_t(int *N, int *NRSH, double complex *A, int *lda, 
	 int *piv, double complex *B, int *ldb, int *info)

cdef sgesv_t *sgesv = <sgesv_t*>f2py_pointer(scipy.linalg.lapack.sgesv._cpointer)
cdef dgesv_t *dgesv = <dgesv_t*>f2py_pointer(scipy.linalg.lapack.dgesv._cpointer)
cdef cgesv_t *cgesv = <cgesv_t*>f2py_pointer(scipy.linalg.lapack.cgesv._cpointer)
cdef zgesv_t *zgesv = <zgesv_t*>f2py_pointer(scipy.linalg.lapack.zgesv._cpointer)

cdef gesv_opt(int A0, int A1, int B0, int B1):
     if A0 != A1:
         raise ValueError('Matrix A is not square for gesv operation')
     return A0, B1

cdef c_sgesv(float[::1,:] A, int[:] pvt, float[::1,:] B):
     
     cdef int n, nrsh, info

     n , nrsh = gesv_opt(A.shape[0],A.shape[1],B.shape[0],B.shape[1])
     sgesv(&n, &nrsh, &A[0,0], &n, &pvt[0], &B[0,0], &n, &info)

cdef c_dgesv(double[::1,:] A, int[:] pvt, double[::1,:] B):
     
     cdef int n, nrsh, info

     n , nrsh = gesv_opt(A.shape[0],A.shape[1],B.shape[0],B.shape[1])
     dgesv(&n, &nrsh, &A[0,0], &n, &pvt[0], &B[0,0], &n, &info)

cdef c_cgesv(float complex[::1,:] A, int[:] pvt, float complex[::1,:] B):
     
     cdef int n, nrsh, info

     n , nrsh = gesv_opt(A.shape[0],A.shape[1],B.shape[0],B.shape[1])
     cgesv(&n, &nrsh, &A[0,0], &n, &pvt[0], &B[0,0], &n, &info)

cdef c_zgesv(double complex[::1,:] A, int[:] pvt, double complex[::1,:] B):
     
     cdef int n, nrsh, info

     n , nrsh = gesv_opt(A.shape[0],A.shape[1],B.shape[0],B.shape[1])
     zgesv(&n, &nrsh, &A[0,0], &n, &pvt[0], &B[0,0], &n, &info)
