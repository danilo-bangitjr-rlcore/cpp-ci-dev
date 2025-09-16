from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from lib_sql.writers.buffered_sql_writer import BufferedSqlWriter
from lib_sql.writers.sql_writer import SqlWriter


def wait_for_event(pred: Callable[[], bool], interval: float, timeout: float):
    """Wait for a predicate to become true within a timeout period."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if pred(): return True
        time.sleep(interval)
    return False


class DummyWriter(SqlWriter[int]):
    def __init__(self):
        self.received: list[int] = []
        self.closed = False

    def write_many(self, rows: Sequence[int]):
        self.received.extend(rows)

    def write(self, row: int):
        self.received.append(row)

    def close(self):
        self.closed = True

    def flush(self):
        ...


def test_soft_flush():
    inner = DummyWriter()
    buf = BufferedSqlWriter(inner, low_watermark=2, high_watermark=4)

    buf.write(1)
    buf.write(2)  # triggers soft flush (background)

    # Wait for background sync to complete
    wait_for_event(
        pred=lambda: inner.received == [1, 2],
        interval=0.01,
        timeout=1.0,
    )
    assert inner.received == [1, 2]


def test_hard_flush_blocks():
    inner = DummyWriter()
    buf = BufferedSqlWriter(inner, low_watermark=2, high_watermark=3)

    buf.write_many([1, 2, 3, 4])  # exceeds high watermark forces blocking
    assert inner.received == [1, 2, 3, 4]


def test_close_flushes_and_closes():
    inner = DummyWriter()
    buf = BufferedSqlWriter(inner, low_watermark=10, high_watermark=20)

    buf.write_many([1, 2, 3])
    buf.close()

    assert inner.received == [1, 2, 3]
    assert inner.closed is True
