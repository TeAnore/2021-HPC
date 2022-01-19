from functools import partial
from timeit import timeit

import numpy as np
import pandas as pd
from CythonWrapper import cuda_matmul, cpu_matmul, cpu_parallel_matmul
from tqdm import tqdm

from CpuMatmul import parallel_matmul, vector_numpy_matmul, single_thread_matmul
from GpuMatmul import gpu_pytorch_matmul, gpu_pycuda_matmul, gpu_pycuda_cublas


def run_benchmark(skip_slow_implementations=False):
    """
        Launch a becnhamrk which measures average executing time
        for all implpemented matrix multiplication methods
    """
    array_size = [2 ** i for i in range(3, 12)]
    repeats = 100
    results = {
        "vector_numpy_matmul": [],
        "gpu_pytorch_matmul": [],
        "gpu_pycuda_matmul": [],
        "gpu_pycuda_cublas": [],
        "cuda_matmul": [],

    }
    if not skip_slow_implementations:
        results['parallel_matmul'] = []
        results['single_thread_matmul'] = []
        results['cpu_matmul'] = []
        results['cpu_parallel_matmul'] = []

    for N in tqdm(array_size[::]):
        a = np.random.randn(N, N).astype(np.float32)
        b = np.random.randn(N, N).astype(np.float32)
        c = np.zeros((N, N), np.float32)

        # cuBLAS uses FORTRAN-like arrays ordering
        a_F = np.random.randn(N, N).astype(np.float32, order="F")
        b_F = np.random.randn(N, N).astype(np.float32, order="F")
        c_F = np.zeros((N, N), dtype=np.float32, order='F')

        if not skip_slow_implementations and N <= 512:
            results['parallel_matmul'] \
                .append(timeit(partial(parallel_matmul, a, b, c, num_proc=4), number=2) / 2)
            results['single_thread_matmul'] \
                .append(timeit(partial(single_thread_matmul, a, b, c), number=2) / 2)
            results['cpu_matmul'] \
                .append(timeit(partial(cpu_matmul, a, b, c), number=2) / 2)
            results['cpu_parallel_matmul'] \
                .append(timeit(partial(cpu_parallel_matmul, a, b, c, n_threads=4), number=2) / 2)
        else:
            results['parallel_matmul'].append(0.0)
            results['single_thread_matmul'].append(0.0)
            results['cpu_matmul'].append(0.0)
            results['cpu_parallel_matmul'].append(0.0)

        results['vector_numpy_matmul'] \
            .append(timeit(partial(vector_numpy_matmul, a, b, c), number=repeats) / repeats)
        results['gpu_pytorch_matmul'] \
            .append(timeit(partial(gpu_pytorch_matmul, a, b, c), number=repeats) / repeats)
        results['gpu_pycuda_matmul'] \
            .append(timeit(partial(gpu_pycuda_matmul, a, b, c), number=repeats) / repeats)
        results['gpu_pycuda_cublas'] \
            .append(timeit(partial(gpu_pycuda_cublas, a_F, b_F, c_F), number=repeats) / repeats)
        results['cuda_matmul'] \
            .append(timeit(partial(cuda_matmul, a_F, b_F, c_F), number=repeats) / repeats)


    df = pd.DataFrame(results)
    df.to_csv("results.csv")


def reset_result(res):
    res[:] = np.zeros(res.shape, res.dtype)


def perform_test(method, a, b, res, GT_result, **kwargs):
    """
        Checks method results equals to ground-truth result,
        otherwise raise AssertionError
    """
    method(a, b, res, **kwargs)
    assert np.allclose(GT_result, res), method.__name__ 
    reset_result(res)

def check_matmul_correctness():
    """
        Performs sainity-check of implemented methods in the way
        comparison implementation resuls with np.dot result
    """
    a_order_f = np.random.randint(0, 10, (8, 10)).astype(np.float32, order='F')
    b_order_f = np.random.randint(0, 10, (10, 7)).astype(np.float32, order='F')
    a = a_order_f.astype(np.float32, order='C', copy=False)
    b = b_order_f.astype(np.float32, order='C', copy=False)

    m = a.shape[0]
    k = b.shape[1]

    res = np.zeros((m, k), dtype=np.float32, order='C')
    res_F = np.zeros((m, k), dtype=np.float32, order='F')

    GT_result = np.dot(a, b)

    perform_test(parallel_matmul, a, b, res, GT_result)
    perform_test(vector_numpy_matmul, a, b, res, GT_result)
    perform_test(single_thread_matmul, a, b, res, GT_result)

    perform_test(gpu_pytorch_matmul, a, b, res, GT_result)
    perform_test(gpu_pycuda_matmul, a, b, res, GT_result)
    perform_test(gpu_pycuda_cublas, a_order_f, b_order_f, res, GT_result)

    perform_test(cuda_matmul, a_order_f, b_order_f, res_F, GT_result)
    perform_test(cpu_matmul, a, b, res, GT_result)
    perform_test(cpu_parallel_matmul, a, b, res, GT_result, n_threads=4)


if __name__ == "__main__":
    check_matmul_correctness()
    print("All methods implemented correctly, results is fine!")
    run_benchmark(skip_slow_implementations=False)
