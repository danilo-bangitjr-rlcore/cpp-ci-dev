from collections.abc import Iterable


class ReplayStorage[T]:
    def __init__(self, capacity: int):
        self.size = 0

        self._capacity = capacity
        self._pos = 0
        self._buffer: list[T | None] = [None] * capacity


    def add(self, item: T):
        idx = self._pos
        self._buffer[idx] = item
        self._pos = (self._pos + 1) % self._capacity

        self.size = min(self.size + 1, self._capacity)

        return idx


    def get_batch(self, idxs: Iterable[int]) -> list[T]:
        out: list[T] = []
        for idx in idxs:
            item = self._buffer[idx]
            assert item is not None
            out.append(item)

        return out


    def last_idx(self):
        assert self.size > 0
        return (self._pos - 1) % self._capacity
