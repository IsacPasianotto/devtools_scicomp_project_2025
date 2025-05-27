"""Code to compile AOT with numba the matmatmul function for performance."""
import numpy as np
from numba import njit, prange
from numba.pycc import CC


cc= CC('numba_compiled_matmul')

dtypes = ['uint8', 'uint16', 'uint32', 'uint64',
          'int8', 'int16', 'int32', 'int64',
          'float32', 'float64']

def register_matmul(dtype):
    sig = f"{dtype}[:,:]({dtype}[:,:], {dtype}[:,:])"

    @cc.export(f"matmul_numbaaot_{dtype}", sig)
    @njit(parallel=True)
    def matmul(A, B):
        n, m = A.shape
        _, p = B.shape
        C = np.zeros((n, p), dtype=A.dtype)
        for i in prange(n):
            for j in range(p):
                for k in range(m):
                    C[i, j] += A[i, k] * B[k, j]
        return C

# Register all variants
for dtype in dtypes:
    register_matmul(dtype)

if __name__ == "__main__":
    cc.compile()
