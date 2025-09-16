import time
from collections.abc import Callable, Sequence
from typing import NamedTuple

import pytest

from lib_sql.writers.point_collecting_sql_writer import PointCollectingSqlWriter
from lib_sql.writers.sql_writer import SqlWriter


class MetricRow(NamedTuple):
    accuracy: float
    loss: float
    epoch: int


class DummyWriter(SqlWriter[MetricRow]):
    def __init__(self):
        self.received: list[MetricRow] = []
        self.closed = False

    def write_many(self, rows: Sequence[MetricRow]) -> None:
        self.received.extend(rows)

    def write(self, row: MetricRow) -> None:
        self.received.append(row)

    def flush(self) -> None:
        ...

    def close(self) -> None:
        self.closed = True


def create_row_factory(default_values: dict[str, float] | None = None) -> Callable[[dict[str, float]], MetricRow]:
    defaults = default_values or {}

    def row_factory(data: dict[str, float]) -> MetricRow:
        merged = {**defaults, **data}
        return MetricRow(
            accuracy=merged.get("accuracy", 0.0),
            loss=merged.get("loss", 0.0),
            epoch=int(merged.get("epoch", 0)),
        )
    return row_factory


@pytest.mark.timeout(5)
def test_basic_write_point_and_flush():
    """
    Test basic write_point and flush operations.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory)

    writer.write_point("accuracy", 0.95)
    writer.write_point("loss", 0.05)
    writer.write_point("epoch", 10)

    assert len(inner.received) == 0

    writer.flush()

    assert len(inner.received) == 1
    row = inner.received[0]
    assert row.accuracy == 0.95
    assert row.loss == 0.05
    assert row.epoch == 10


@pytest.mark.timeout(5)
def test_empty_flush():
    """
    Test flushing with no collected points is a no-op.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory)

    writer.flush()
    assert len(inner.received) == 0


@pytest.mark.timeout(5)
def test_metric_overwriting():
    """
    Test writing to same metric overwrites previous value.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory)

    writer.write_point("accuracy", 0.8)
    writer.write_point("accuracy", 0.95)
    writer.write_point("loss", 0.05)

    writer.flush()

    assert len(inner.received) == 1
    row = inner.received[0]
    assert row.accuracy == 0.95
    assert row.loss == 0.05


@pytest.mark.timeout(5)
def test_multiple_flush_cycles():
    """
    Test multiple collect-flush cycles with storage clearing.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory)

    writer.write_point("accuracy", 0.9)
    writer.write_point("loss", 0.1)
    writer.flush()

    writer.write_point("accuracy", 0.95)
    writer.write_point("epoch", 5)
    writer.flush()

    assert len(inner.received) == 2
    assert inner.received[0].accuracy == 0.9
    assert inner.received[0].loss == 0.1
    assert inner.received[0].epoch == 0

    assert inner.received[1].accuracy == 0.95
    assert inner.received[1].loss == 0.0
    assert inner.received[1].epoch == 5


@pytest.mark.timeout(5)
def test_row_factory_with_defaults():
    """
    Test row factory with default values for missing metrics.
    """
    inner = DummyWriter()
    default_values = {"accuracy": 0.5, "loss": 1.0, "epoch": 1}
    row_factory = create_row_factory(default_values)
    writer = PointCollectingSqlWriter(inner, row_factory)

    writer.write_point("accuracy", 0.9)
    writer.flush()

    assert len(inner.received) == 1
    row = inner.received[0]
    assert row.accuracy == 0.9
    assert row.loss == 1.0
    assert row.epoch == 1


@pytest.mark.timeout(5)
def test_partial_metrics_collection():
    """
    Test collecting only some of the expected metrics.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory)

    writer.write_point("loss", 0.3)
    writer.flush()

    assert len(inner.received) == 1
    row = inner.received[0]
    assert row.accuracy == 0.0
    assert row.loss == 0.3
    assert row.epoch == 0


@pytest.mark.timeout(5)
def test_non_numeric_values():
    """
    Test float values converted to int by row factory.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory)

    writer.write_point("accuracy", 0.95)
    writer.write_point("loss", 0.05)
    writer.write_point("epoch", 15.7)

    writer.flush()

    assert len(inner.received) == 1
    row = inner.received[0]
    assert row.accuracy == 0.95
    assert row.loss == 0.05
    assert row.epoch == 15


@pytest.mark.timeout(5)
def test_disabled_writer():
    """
    Test disabled writer ignores all operations.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory, enabled=False)

    writer.write_point("accuracy", 0.95)
    writer.write_point("loss", 0.05)
    writer.flush()

    assert len(inner.received) == 0


@pytest.mark.timeout(5)
def test_enabled_writer_explicit():
    """
    Test explicitly enabled writer works normally.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory, enabled=True)

    writer.write_point("accuracy", 0.95)
    writer.flush()

    assert len(inner.received) == 1
    assert inner.received[0].accuracy == 0.95


@pytest.mark.timeout(5)
def test_close_flushes_and_delegates():
    """
    Test close() flushes pending data and closes inner writer.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory)

    writer.write_point("accuracy", 0.9)
    writer.write_point("loss", 0.1)

    writer.close()

    assert len(inner.received) == 1
    assert inner.received[0].accuracy == 0.9
    assert inner.received[0].loss == 0.1
    assert inner.closed is True


@pytest.mark.timeout(5)
def test_close_with_empty_storage():
    """
    Test close() with no collected data still delegates to inner writer.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory)

    writer.close()

    assert len(inner.received) == 0
    assert inner.closed is True


@pytest.mark.timeout(5)
def test_close_when_disabled():
    """
    Test close() behavior when writer is disabled.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory, enabled=False)

    writer.write_point("accuracy", 0.9)
    writer.close()

    assert len(inner.received) == 0
    assert inner.closed is True


@pytest.mark.timeout(5)
def test_storage_persistence_across_flushes():
    """
    Test storage is properly cleared after each flush.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory)

    writer.write_point("accuracy", 0.8)
    writer.flush()

    writer.write_point("loss", 0.2)
    writer.flush()

    assert len(inner.received) == 2
    assert inner.received[0].accuracy == 0.8
    assert inner.received[0].loss == 0.0
    assert inner.received[0].epoch == 0

    assert inner.received[1].accuracy == 0.0
    assert inner.received[1].loss == 0.2
    assert inner.received[1].epoch == 0


@pytest.mark.timeout(5)
def test_extra_metrics_in_storage():
    """
    Test row factory handles extra metrics not in target row type.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory)

    writer.write_point("accuracy", 0.95)
    writer.write_point("loss", 0.05)
    writer.write_point("f1_score", 0.88)
    writer.write_point("epoch", 5)

    writer.flush()

    assert len(inner.received) == 1
    row = inner.received[0]
    assert row.accuracy == 0.95
    assert row.loss == 0.05
    assert row.epoch == 5


@pytest.mark.timeout(5)
def test_auto_flush_basic():
    """
    Test basic auto-flush functionality with short delay.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory, auto_collect_delay=0.1)

    writer.write_point("accuracy", 0.95)

    assert len(inner.received) == 0

    time.sleep(0.15)

    assert len(inner.received) == 1
    assert inner.received[0].accuracy == 0.95

    writer.close()


@pytest.mark.timeout(5)
def test_auto_flush_single_timer():
    """
    Test only one timer is scheduled with multiple write_point calls.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory, auto_collect_delay=0.2)

    writer.write_point("accuracy", 0.9)
    writer.write_point("loss", 0.1)
    writer.write_point("epoch", 5)

    time.sleep(0.1)
    assert len(inner.received) == 0

    time.sleep(0.15)
    assert len(inner.received) == 1

    row = inner.received[0]
    assert row.accuracy == 0.9
    assert row.loss == 0.1
    assert row.epoch == 5

    writer.close()


@pytest.mark.timeout(5)
def test_auto_flush_timer_not_reset():
    """
    Test timer is not reset when new data is added.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory, auto_collect_delay=0.2)

    writer.write_point("accuracy", 0.9)

    time.sleep(0.1)

    writer.write_point("loss", 0.1)

    time.sleep(0.15)

    assert len(inner.received) == 1
    row = inner.received[0]
    assert row.accuracy == 0.9
    assert row.loss == 0.1

    writer.close()


@pytest.mark.timeout(5)
def test_auto_flush_disabled_when_no_delay():
    """
    Test that auto-collect is disabled when auto_collect_delay is None.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory)  # No auto_collect_delay

    writer.write_point("accuracy", 0.95)

    time.sleep(0.2)

    assert len(inner.received) == 0

    writer.flush()
    assert len(inner.received) == 1

    writer.close()


@pytest.mark.timeout(5)
def test_manual_flush_cancels_auto_flush():
    """
    Test manual flush cancels pending auto-flush timer.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory, auto_collect_delay=0.3)

    writer.write_point("accuracy", 0.9)

    writer.flush()
    assert len(inner.received) == 1

    time.sleep(0.35)

    assert len(inner.received) == 1

    writer.close()


@pytest.mark.timeout(5)
def test_close_cancels_auto_flush_timer():
    """
    Test close() cancels any pending auto-flush timer.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory, auto_collect_delay=0.3)

    writer.write_point("accuracy", 0.9)
    writer.close()

    assert len(inner.received) == 1
    assert inner.closed is True

    time.sleep(0.35)

    assert len(inner.received) == 1


@pytest.mark.timeout(5)
def test_auto_flush_disabled_writer():
    """
    Test auto-flush doesn't work when writer is disabled.
    """
    inner = DummyWriter()
    row_factory = create_row_factory()
    writer = PointCollectingSqlWriter(inner, row_factory, enabled=False, auto_collect_delay=0.1)

    writer.write_point("accuracy", 0.95)

    time.sleep(0.15)

    assert len(inner.received) == 0

    writer.close()
