import ctypes
from functools import partial
from platform import machine
from typing import Tuple
import numpy as np
from typing import Tuple, List
import multiprocessing as mp
from multiprocessing import shared_memory


def single_thread_matmul(a: np.ndarray, b: np.ndarray, res: np.ndarray):
    """
        Python naive implementation of matrix multiplication in single thread,
        Extremly slow! Be aware of using huge matrices
    """
    m, n = a.shape
    _n, k = b.shape
    assert n == _n, "Incorrect matrices shape"

    for m_iter in range(m):
        for k_iter in range(k):
            for n_iter in range(n):
                res[m_iter, k_iter] += a[m_iter, n_iter] * b[n_iter, k_iter]


def parallel_matmul(a: np.ndarray, b: np.ndarray, res: np.ndarray, num_proc: int = 2):
    """
        Python naive matrix multiplication with multiprocessinig.
        Each process computes sub-matrices
        Results saving in shared memory

        P.S. I use exactly multiprocessing due to python GIL 
    """
    m, n = a.shape
    _n, k = b.shape
    assert n == _n, "Incorrect matrices shape"

    shm = shared_memory.SharedMemory(create=True, size=res.nbytes)
    shm_ndarray = np.ndarray(res.shape, dtype=np.float32, buffer=shm.buf)

    part_size = m // num_proc
    with mp.Pool(num_proc) as pool:
        # Since we are on linux new processes inherit all the data from the parent process.
        pool.map(partial(matmul_line, a=a, b=b, shm_name=shm.name, res_shape=(m, k),
                 part_size=part_size, m=m, n=n, k=k), range(0, m, part_size))
    res[:] = shm_ndarray[:]
    shm.close()
    shm.unlink()

def vector_numpy_matmul(a: np.ndarray, b: np.ndarray, res: np.ndarray):
    np.dot(a, b, res)


def matmul_line(idx: int, a: Tuple[np.ndarray, List], b: Tuple[np.ndarray, List], shm_name: str, res_shape: Tuple, part_size: int, m: int, n: int, k: int):
    """
        Computes matrix multiplication for a matrix and save results in shared memory
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    result = np.ndarray(res_shape, dtype=np.float32, buffer=shm.buf)
    for m_iter in range(idx, idx + part_size if idx + part_size < res_shape[0] else res_shape[0]):
        for k_iter in range(k):
            for n_iter in range(n):
                result[m_iter, k_iter] += a[m_iter, n_iter] * b[n_iter, k_iter]
    del result
    shm.close()