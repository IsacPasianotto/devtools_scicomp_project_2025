import pymatmatmul.numba_compiled_matmul as numba_aot
import pymatmatmul.libmatmul as aotmm
from pymatmatmul.utils import setup_logger

import numpy as np

AOT_BACKEND_DISPATCH = dict()
# analyze every function in the compiled namespace
for function_name in dir(aotmm):
    #given a name, get the object
    attr = getattr(aotmm, function_name)
    #look only at the callable
    if callable(attr):
        for d in numba_aot.supported_numba_dtypes:
            if function_name == (f"matmul_numbaaot_{d}"):
                AOT_BACKEND_DISPATCH[np.dtype(d)] = attr
                break

SUPPORTED_DTYPES = list(map(np.dtype,numba_aot.supported_numba_dtypes))

logger = setup_logger()
