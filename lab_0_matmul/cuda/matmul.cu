#include <cublas_v2.h>
#include <iostream>

int GPU_ID = 1;
float *d_A = NULL, *d_B = NULL, *d_C = NULL;
int CACHED_M = -1, CACHED_N = -1, CACHED_K = -1;

cublasHandle_t HANDLE = NULL;


// Useful function for logging CUDA side errors
#define CHECK_ERRORS_DECORATOR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


float* allocate_mem_device(const int n, const int m) {
    size_t mat_size = n * m * sizeof(float);
    float *dev_mat;
    CHECK_ERRORS_DECORATOR(cudaMalloc(&dev_mat, mat_size));
    return dev_mat;
}


void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int n, const int k) {
    int lda = m;
    int ldb = n;
    int ldc = m;
    const float alpha = 1.0;
    const float beta = 0.0;

    if (HANDLE == NULL) {
        cublasCreate(&HANDLE); 
    }
    /*
    C = α op ( A ) op ( B ) + β C 
    handle - input - handle to the cuBLAS library context.
    transa - input - operation op(A) that is non- or (conj.) transpose.
    transb - input - operation op(B) that is non- or (conj.) transpose.
    m - input - number of rows of matrix op(A) and C.
    n - input - number of columns of matrix op(B) and C.
    k - input - number of columns of op(A) and rows of op(B).
    alpha - host or device - input - <type> scalar used for multiplication.
    A - device - input - <type> array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
    lda - input - leading dimension of two-dimensional array used to store the matrix A.
    B - device - input - <type> array of dimension ldb x n with ldb>=max(1,k) if transb == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
    ldb - input - leading dimension of two-dimensional array used to store matrix B.
    beta - host or device - input - <type> scalar used for multiplication. If beta==0, C does not have to be a valid input.
    C - device - in/out - <type> array of dimensions ldc x n with ldc>=max(1,m).
    ldc - input - leading dimension of a two-dimensional array used to store the matrix C. 
    */
    cublasSgemm(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, A, lda, B, ldb, &beta, C, ldc); 
}

extern "C"
float gpu_matmul(const float *A, const float *B, float *C, const int m, const int n, const int k) {
    // cudaSetDevice(1);

    // Preventing memory reallocation
    if ((m != CACHED_M) || (n != CACHED_N) || (k != CACHED_K)) {
        if (d_A != NULL || d_B != NULL || d_C != NULL) {
            CHECK_ERRORS_DECORATOR(cudaFree(d_A));
            CHECK_ERRORS_DECORATOR(cudaFree(d_B));
            CHECK_ERRORS_DECORATOR(cudaFree(d_C));
        }
        CACHED_K = k;
        CACHED_M = m;
        CACHED_N = n;
        d_A = allocate_mem_device(m, n);
        d_B = allocate_mem_device(n, k);
        d_C = allocate_mem_device(m, k);
    }

    size_t size_a = m * n * sizeof(float);
    size_t size_b = n * k * sizeof(float);
    size_t size_c = m * k * sizeof(float);

    CHECK_ERRORS_DECORATOR(cudaMemcpy(d_A, A, size_a, cudaMemcpyHostToDevice)); 
    CHECK_ERRORS_DECORATOR(cudaMemcpy(d_B, B, size_b, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop; 
    float gpuTime = 0.0;

    CHECK_ERRORS_DECORATOR(cudaEventCreate(&start));
    CHECK_ERRORS_DECORATOR(cudaEventCreate(&stop));
    CHECK_ERRORS_DECORATOR(cudaEventRecord(start, 0));
    
    gpu_blas_mmul(d_A, d_B, d_C, m, n, k); 
    
    CHECK_ERRORS_DECORATOR(cudaDeviceSynchronize());
    CHECK_ERRORS_DECORATOR(cudaEventRecord(stop, 0));
        
    CHECK_ERRORS_DECORATOR(cudaMemcpy(C, d_C, size_c, cudaMemcpyDeviceToHost));
    
    CHECK_ERRORS_DECORATOR(cudaEventElapsedTime(&gpuTime, start, stop));

    return gpuTime;
}
