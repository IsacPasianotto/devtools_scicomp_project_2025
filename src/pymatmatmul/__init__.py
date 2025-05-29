import pymatmatmul.numba_compiled_matmul as numba_aot
import pymatmatmul.libmatmul as aotmm
import numpy as np

AOT_BACKEND_DISPATCH = dict()
for name in dir(aotmm):
    attr = getattr(aotmm, name)
    if callable(attr):
        for d in numba_aot.supported_numba_dtypes:
            if name.endswith(d):
                AOT_BACKEND_DISPATCH[np.dtype(d)] = attr
                break

SUPPORTED_DTYPES = list(map(np.dtype,numba_aot.supported_numba_dtypes))