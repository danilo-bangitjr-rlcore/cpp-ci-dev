from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np


class ReplayStorage[T: NamedTuple]:
    def __init__(self, capacity: int):
        self._capacity = capacity

        self._pos = 0
        self._size = 0
        self._data: tuple[np.ndarray, ...] | None = None
        # type checker won't be able to infer this type correctly
        self._tuple_builder: Any | None = None


    def add(self, item: T):
        if self._data is None:
            self._data = self._init(item)

        idx = self._pos

        for element, buffer in zip(item, self._data, strict=True):
            buffer[idx] = element

        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
        return idx


    def _init(self, item: T):
        elements = tuple(
            element if isinstance(element, np.ndarray) else np.asarray(element)
            for element in item
        )

        buffers = tuple(
            np.empty((self._capacity, *element.shape), element.dtype)
            for element in elements
        )

        self._data = buffers
        self._tuple_builder = type(item)

        return self._data

    def get_ensemble_batch(self, idxs: list[np.ndarray]) -> T:
        assert self._data is not None
        assert self._tuple_builder is not None

        return self._tuple_builder(*(
            jnp.stack([buffer[sub_idxs] for sub_idxs in idxs], axis=0)
            for buffer in self._data
        ))


    def get_batch(self, idxs: np.ndarray) -> T:
        assert self._data is not None
        assert self._tuple_builder is not None
        return self._tuple_builder(
            *(jnp.asarray(buffer[idxs]) for buffer in self._data),
        )


    def last_idx(self):
        assert self._size > 0
        return (self._pos - 1) % self._capacity


    def last_idxs(self, n: int) -> np.ndarray:
        assert self._size > 0

        if self._size < self._capacity:
            n = min(n, self._size)
            start = (self._pos - n) % self._size
            return np.arange(start, self._pos)

        return np.arange(self._pos - n, self._pos) % self._capacity


    def size(self):
        return self._size
