import time

import callback  # pyright: ignore [reportMissingImports]
import numpy as np

print(callback.foo.__doc__)


# def f(i): return i*i
#
# print(callback.foo(f))
def wait():
    print("called")
    time.sleep(1)
    return np.asarray([1,2], dtype=np.float64)

print(callback.foo(wait))
