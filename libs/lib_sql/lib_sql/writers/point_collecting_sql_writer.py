from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from lib_sql.writers.sql_writer import SqlWriter


class PointCollectingSqlWriter[T]:
    """Point-based metric collection writer.

    Collects individual metric-value pairs in an internal storage dict until
    flush() is called, then writes a complete row to the underlying SqlWriter.
    Requires a row_factory function to convert the collected dict to type T.
    """

    def __init__(
        self,
        inner: SqlWriter[T],
        row_factory: Callable[[dict[str, Any]], T],
        *,
        enabled: bool = True,
        auto_flush_delay: float | None = None,
    ):
        self._inner = inner
        self._row_factory = row_factory
        self._enabled = enabled
        self._auto_flush_delay = auto_flush_delay
        self._storage: dict[str, Any] = {}
        self._auto_flush_timer: threading.Timer | None = None

    def write_point(self, metric: str, value: float) -> None:
        if not self._enabled:
            return

        self._storage[metric] = value
        self._schedule_auto_flush()

    def flush(self) -> None:
        """Flush collected metrics to the underlying SqlWriter as a single row.

        Constructs a row object from the current storage dict using the row_factory
        and writes it using the inner SqlWriter's write() method. Clears storage after writing.
        """
        if not self._enabled or not self._storage:
            return

        self._cancel_auto_flush_timer()

        row = self._row_factory(self._storage)
        self._inner.write(row)

        self._storage.clear()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._cancel_auto_flush_timer()
            self._inner.close()

    def _schedule_auto_flush(self) -> None:
        if not self._enabled or self._auto_flush_delay is None:
            return

        if self._auto_flush_timer is None or not self._auto_flush_timer.is_alive():
            self._auto_flush_timer = threading.Timer(self._auto_flush_delay, self.flush)
            self._auto_flush_timer.start()

    def _cancel_auto_flush_timer(self) -> None:
        if self._auto_flush_timer is not None and self._auto_flush_timer.is_alive():
            self._auto_flush_timer.cancel()
            self._auto_flush_timer = None
