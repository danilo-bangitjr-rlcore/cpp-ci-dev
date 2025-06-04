from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u


class BufferState(NamedTuple):
    pos: int
    size: jax.Array
    capacity: int
    data: tuple[jax.Array, ...]


class ReplayStorage[T: NamedTuple]:
    def __init__(self, capacity: int):
        self._capacity = capacity

        self._state: BufferState | None = None
        # type checker won't be able to infer this type correctly
        self._tuple_builder: Any | None = None


    def add(self, item: T):
        if self._state is None:
            self._state = self._init(item)
            self._tuple_builder = type(item)

        idx, self._state = self._add(self._state, item)
        return idx

    @jax_u.method_jit
    def _add(self, state: BufferState, item: T):
        idx = state.pos

        new_data = tuple(
            buffer.at[idx].set(element)
            for buffer, element in zip(state.data, item, strict=True)
        )

        pos = (state.pos + 1) % state.capacity

        return idx, BufferState(
            pos=pos,
            size=jnp.minimum(state.size + 1, state.capacity),
            capacity=state.capacity,
            data=new_data,
        )


    def _init(self, item: T):
        elements = tuple(
            element if isinstance(element, jax.Array) else jnp.array(element)
            for element in item
        )

        buffers = tuple(
            jnp.empty((self._capacity, *element.shape), element.dtype)
            for element in elements
        )

        return BufferState(
            pos=0,
            size=jnp.zeros(1, dtype=jnp.int32),
            capacity=self._capacity,
            data=buffers,
        )

    def get_ensemble_batch(self, idxs: list[jax.Array]) -> T:
        assert self._state is not None
        assert self._tuple_builder is not None
        raw_buffers = self._get_ensemble_batch(self._state, idxs)
        return self._tuple_builder(*raw_buffers)


    @jax_u.method_jit
    def _get_ensemble_batch(self, state: BufferState, idxs: list[jax.Array]):
        return tuple(
            jnp.stack([buffer[sub_idxs] for sub_idxs in idxs], axis=0)
            for buffer in state.data
        )


    def get_batch(self, idxs: jax.Array) -> T:
        assert self._state is not None
        assert self._tuple_builder is not None
        raw_buffers = self._get_batch(self._state, idxs)
        return self._tuple_builder(*raw_buffers)


    @jax_u.method_jit
    def _get_batch(self, state: BufferState, idxs: jax.Array):
        return tuple(buffer[idxs] for buffer in state.data)


    def last_idx(self):
        assert self._state is not None
        assert self._state.size.item() > 0
        return (self._state.pos - 1) % self._capacity


    def size(self):
        if self._state is None:
            return 0

        return int(self._state.size.item())
