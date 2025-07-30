from typing import Any, Generic, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten

# NOTE: the python 3.12+ syntax for generic types is not compatible with pickle
T = TypeVar('T', bound=NamedTuple)


class ReplayStorage(Generic[T]): # noqa: UP046
    def __init__(self, capacity: int):
        self._capacity = capacity

        self._pos = 0
        self._size = 0
        self._data: tuple[np.ndarray, ...] | None = None
        # type checker won't be able to infer this type correctly
        self._treedef: Any


    def add(self, item: T):
        if self._data is None:
            self._data = self._init(item)

        idx = self._pos

        leaves, _ = tree_flatten(item)
        for leaf, buffer in zip(leaves, self._data, strict=True):
            buffer[idx] = np.asarray(leaf)

        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
        return idx


    def _init(self, item: T):
        try:
            leaves, treedef = tree_flatten(item)
            for leaf in leaves: assert isinstance(leaf, jax.Array | int | float)
            self._treedef = treedef
        except(Exception) as e:
            raise NotImplementedError(
                "storage backend only supports pytrees with leaves of type: jax.Array | int | float",
            ) from e

        elements = tuple(np.asarray(leaf) for leaf in leaves)
        buffers = tuple(
            np.empty((self._capacity, *element.shape), element.dtype)
            for element in elements
        )

        self._data = buffers
        return self._data

    def get_ensemble_batch(self, idxs: list[np.ndarray]) -> T:
        assert self._data is not None
        return tree_unflatten(
            self._treedef,
            (jnp.stack([buffer[sub_idxs] for sub_idxs in idxs], axis=0) for buffer in self._data),
        )

    def get_batch(self, idxs: np.ndarray) -> T:
        assert self._data is not None
        return tree_unflatten(
            self._treedef,
            (jnp.asarray(buffer[idxs]) for buffer in self._data),
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
