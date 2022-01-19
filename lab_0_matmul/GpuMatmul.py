from linecache import cache
from time import time
from typing import List, Tuple

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import skcuda.linalg as linalg
import torch
from pycuda import compiler, gpuarray

# Required by scikit-cuda to initialize global cublas handler


pycuda_kernel_matmul = """
    __global__ void MatMulKernel(const float *a, const float *b, float *c, const int M, const int N, const int K)
    {
        int tx = threadIdx.x + blockIdx.x * blockDim.x;
        int ty = threadIdx.y + blockIdx.y * blockDim.y;
        
        float sum = 0;

        for (int i = 0; i < N; i++) {
            sum += a[ty * N + i] * b[i * K + tx];
        }

        c[ty * K + tx] = sum;
    }
"""


# Variables used to prevent reallocation memory on GPU
a_D = b_D = c_D = None
cached_m = cached_n = cached_k = None



def gpu_pytorch_matmul(a: np.ndarray, b: np.ndarray, res: np.ndarray, device: str = "cuda:0"):
    """
        Pytorch gpu implementation of matrix multiplication
    """
    a = torch.from_numpy(a).to(device)
    b = torch.from_numpy(b).to(device)
    d_res = torch.zeros(res.shape).to(device)

    torch.matmul(a, b, out=d_res)
    res[:] = d_res.detach().cpu().numpy()


def gpu_pycuda_cublas(a: np.ndarray, b: np.ndarray, res: np.ndarray):
    """
        PyCuda Cublas (scikit-cuda) implementation of matrix multiplication
    """
    if not (a.data.f_contiguous and b.data.f_contiguous):
        raise RuntimeError("For using methods based on CUBLAS library you should provied an" +
            " arrays in FORTRAN! order")
    linalg.init()

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = linalg.dot(a_gpu, b_gpu)
    res[:] = c_gpu.get()



def gpu_pycuda_matmul(a: np.ndarray, b: np.ndarray, res: np.ndarray):
    """
        PyCuda with cuda kerlen implementation
    """
    global cached_m, cached_n, cached_k, a_D, b_D, c_D
    # Check if memory already allocated and have suitable shape for the matrieces
    if (cached_m != a.shape[0]) or (cached_n != a.shape[1] or (cached_k != b.shape[1])):
        if a_D is not None:
            a_D.free()
            b_D.free()
            c_D.free()

        a_D = cuda.mem_alloc(a.nbytes)
        b_D = cuda.mem_alloc(b.nbytes)
        c_D = cuda.mem_alloc(res.nbytes)

        cached_m = a.shape[0]
        cached_n = a.shape[1]
        cached_k = b.shape[1]

    cuda.memcpy_htod(a_D, a.astype(np.float32))
    cuda.memcpy_htod(b_D, b.astype(np.float32))

    # Compile cuda kerlen by PyCuda
    # If I understood corectly - if kerlen already compiled the method won't recompile it
    # Cannot make it global (like cache) due to PyCuda missing a context
    compiled_kernel = compiler.SourceModule(pycuda_kernel_matmul)
    matmul = compiled_kernel.get_function("MatMulKernel")
    
    # Due to i generate only 2**N size matrices it in the optimal way
    THREADS_IN_BLOCK = 32
    if cached_k > THREADS_IN_BLOCK or cached_m > THREADS_IN_BLOCK:
        dx, mx = divmod(cached_k, THREADS_IN_BLOCK)
        dy, my = divmod(cached_m, THREADS_IN_BLOCK)

        # Number of threads in block
        bdim = (THREADS_IN_BLOCK, THREADS_IN_BLOCK, 1)
        # Number of blocks in grid
        gdim = (dx + (mx>0), dy + (my>0))
    else:
        bdim = (cached_k, cached_m, 1)
        gdim = (1, 1)
        
    matmul(a_D, b_D, c_D, np.int32(cached_m), np.int32(cached_n), np.int32(cached_k), block=bdim, grid=gdim)
    
    cuda.memcpy_dtoh(res, c_D)


