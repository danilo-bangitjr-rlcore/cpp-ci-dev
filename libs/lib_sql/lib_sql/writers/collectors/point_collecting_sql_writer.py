import threading
from collections.abc import Callable
from typing import Any

from lib_sql.writers.sql_writer import SqlWriter


class PointCollectingSqlWriter[T]:
    """Point-based metric collection writer.

    Collects individual metric-value pairs in an internal storage dict until
    collect_row() is called, then writes a complete row to the underlying SqlWriter.
    Requires a row_factory function to convert the collected dict to type T.
    """

    def __init__(
        self,
        inner: SqlWriter[T],
        row_factory: Callable[[dict[str, Any]], T],
        *,
        enabled: bool = True,
        auto_collect_delay: float | None = None,
    ):
        self._inner = inner
        self._row_factory = row_factory
        self._enabled = enabled
        self._auto_collect_delay = auto_collect_delay
        self._storage: dict[str, Any] = {}
        self._auto_collect_timer: threading.Timer | None = None

    def write_point(self, metric: str, value: Any) -> None:
        if not self._enabled:
            return

        self._storage[metric] = value
        self._schedule_auto_collect()

    def collect_row(self) -> None:
        if not self._enabled or not self._storage:
            return

        self._cancel_auto_collect_timer()

        row = self._row_factory(self._storage)
        self._inner.write(row)

        self._storage.clear()

    def flush(self) -> None:
        self.collect_row()
        self._inner.flush()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._cancel_auto_collect_timer()
            self._inner.close()

    def _schedule_auto_collect(self) -> None:
        if not self._enabled or self._auto_collect_delay is None:
            return

        if self._auto_collect_timer is None or not self._auto_collect_timer.is_alive():
            self._auto_collect_timer = threading.Timer(self._auto_collect_delay, self.collect_row)
            self._auto_collect_timer.start()

    def _cancel_auto_collect_timer(self) -> None:
        if self._auto_collect_timer is not None and self._auto_collect_timer.is_alive():
            self._auto_collect_timer.cancel()
            self._auto_collect_timer = None
