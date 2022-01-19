#cython: language_level=3
cimport cython
from cython.parallel cimport prange, parallel


# Define function from static library
cdef extern from "cuda/matmul.h":
    float gpu_matmul(const float *A,  const float *B, float *C, const int m, const int n, const int k)


# Python wrapper for C function
def cuda_matmul(float[:, :] a, float[:, :] b, float[:, :] c):
    cdef:
        int m = a.shape[0]
        int n = a.shape[1]
        int k = b.shape[1]
    return gpu_matmul(&a[0, 0], &b[0, 0], &c[0, 0], m, n, k)

# Disable all checks to increase perfomance
@cython.overflowcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def cpu_matmul(float[:, :] a, float[:, :] b, float[:, :] c):
    cdef:
        int i, j, k
        float res
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            res = 0.0
            for k in range(a.shape[1]):
                res = res + a[i, k] * b[k, j]
            c[i, j] = res


@cython.overflowcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def cpu_parallel_matmul(float[:, :] a, float[:, :] b, float[:, :] c, int n_threads):
    cdef:
        int i, j, k
        float res
    # parallel and prange - OpenMP functions
    with nogil, parallel(num_threads=n_threads):
        for i in prange(a.shape[0], schedule='static'):
            for j in range(b.shape[1]):
                res = 0.0
                for k in range(a.shape[1]):
                    res = res + a[i, k] * b[k, j]
                c[i, j] = res
