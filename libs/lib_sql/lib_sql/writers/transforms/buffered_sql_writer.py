from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

from lib_sql.writers.sql_writer import SqlWriter


class BufferedSqlWriter[T]:
    """Watermark-based buffered writer.

    Wraps a concrete SqlWriter implementation and batches writes based on
    configurable low/high watermarks. Background flush uses a single worker
    thread; a hard watermark forces synchronous wait & immediate reschedule.
    """

    def __init__(
        self,
        inner: SqlWriter[T],
        *,
        low_watermark: int = 1,
        high_watermark: int = 20_000,
        enabled: bool = True,
    ):
        if high_watermark <= 0 or low_watermark <= 0:
            raise ValueError("Watermarks must be positive")
        if high_watermark <= low_watermark:
            raise ValueError("high_watermark must be > low_watermark")

        self._inner = inner
        self._low_wm = low_watermark
        self._high_wm = high_watermark
        self._enabled = enabled

        self._buf: list[T] = []
        self._exec = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._last_sync_time = time.time()

    # ----------------- internal state helpers -----------------
    def __len__(self):
        return len(self._buf)

    def _is_writing(self):
        return self._future is not None and not self._future.done()

    # ----------------- public api -----------------
    def write_many(self, rows: Sequence[T]):
        if not self._enabled or not rows:
            return

        self._buf.extend(rows)

        if len(self._buf) >= self._high_wm:
            self._hard_sync()

        elif len(self._buf) >= self._low_wm:
            self._background_sync()

    def write(self, row: T):
        self.write_many([row])

    def flush(self):
        if not self._enabled:
            return

        if self._future is not None:
            self._future.result()

        self._background_sync()

        if self._future is not None:
            self._future.result()

        # Forward flush to inner writer
        self._inner.flush()

    def close(self):
        try:
            self.flush()
        finally:
            self._exec.shutdown()
            self._inner.close()

    # ----------------- sync operations -----------------
    def _hard_sync(self):
        if self._future is not None:
            self._future.result()

        self._background_sync()

        if self._future is not None:
            self._future.result()

    def _background_sync(self):
        if not self._enabled or self._is_writing() or not self._buf:
            return

        data = self._buf
        self._buf = []
        self._last_sync_time = time.time()
        self._future = self._exec.submit(self._deferred_write, data)

    def _deferred_write(self, batch: Iterable[T]):
        self._inner.write_many(list(batch))

        if len(self._buf) >= self._low_wm:
            self._future = self._exec.submit(self._background_sync)
