from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.engine import Engine

from lib_sql.inspection import table_count, table_size_mb
from lib_sql.utils import SQLColumn, create_tsdb_table_query
from lib_sql.writers.core.dynamic_schema_sql_writer import DynamicSchemaSqlWriter
from lib_sql.writers.core.normalized_narrow_sql_writer import NormalizedNarrowSqlWriter
from lib_sql.writers.core.static_schema_sql_writer import StaticSchemaSqlWriter
from lib_sql.writers.sql_writer import SqlWriter

pytest_plugins = [
    "test.infrastructure.networking",
    "test.infrastructure.utils.docker",
    "test.infrastructure.utils.tsdb",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _timings_log_path() -> Path:
    return _repo_root() / "docs" / "working" / "timings.txt"


def _scenario_seed(*, namespace: str, num_rows: int, num_metrics: int) -> int:
    digest = hashlib.sha1(
        f"{namespace}:{num_rows}:{num_metrics}".encode(),
        usedforsecurity=False,
    ).hexdigest()
    return int(digest[:16], 16)


def _append_benchmark_result(*, scenario: str, writer: str, rows: int, duration: float, size_mb: float) -> None:
    log_path = _timings_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now(UTC).isoformat()} scenario={scenario} writer={writer} "
            f"rows={rows} duration_seconds={duration:.6f} size_mb={size_mb:.2f}\n",
        )


@pytest.fixture(scope="session", autouse=True)
def _reset_benchmark_timings_file() -> Iterator[None]:  # type: ignore[reportUnusedFunction]
    log_path = _timings_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    log_path.touch()
    yield


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    num_rows: int
    num_metrics: int


@dataclass(frozen=True)
class BenchmarkOutcomes:
    max_latency_seconds: float
    max_size_mb: float


@dataclass(frozen=True)
class WriterBenchmarkConfig:
    name: str
    create_writer: Callable[[Engine, str], SqlWriter]
    generate_rows: Callable[[int, int], list[dict[str, Any]]]
    row_count_fn: Callable[[Engine, SqlWriter], int]
    table_refs_fn: Callable[[SqlWriter], Sequence[tuple[str | None, str]]]


SCENARIOS: Sequence[BenchmarkScenario] = (
    BenchmarkScenario(name="r100_m20", num_rows=100, num_metrics=20),
    BenchmarkScenario(name="r100_m100", num_rows=100, num_metrics=100),
    BenchmarkScenario(name="r100_m500", num_rows=100, num_metrics=500),

    BenchmarkScenario(name="r500_m50", num_rows=500, num_metrics=50),
    BenchmarkScenario(name="r1000_m50", num_rows=1_000, num_metrics=50),
    BenchmarkScenario(name="r2000_m50", num_rows=2_000, num_metrics=50),
)


EXPECTED_OUTCOMES: dict[str, dict[str, BenchmarkOutcomes]] = {
    # Buffer of 2x time and 10% size (in MB)
    "r100_m20": {
        "dynamic_schema": BenchmarkOutcomes(max_latency_seconds=0.14, max_size_mb=0.08),
        "normalized_narrow": BenchmarkOutcomes(max_latency_seconds=0.20, max_size_mb=0.30),
        "static_schema": BenchmarkOutcomes(max_latency_seconds=0.63, max_size_mb=0.29),
    },
    "r100_m100": {
        "dynamic_schema": BenchmarkOutcomes(max_latency_seconds=0.22, max_size_mb=0.16),
        "normalized_narrow": BenchmarkOutcomes(max_latency_seconds=0.57, max_size_mb=0.88),
        "static_schema": BenchmarkOutcomes(max_latency_seconds=2.86, max_size_mb=0.94),
    },
    "r100_m500": {
        "dynamic_schema": BenchmarkOutcomes(max_latency_seconds=0.86, max_size_mb=0.49),
        "normalized_narrow": BenchmarkOutcomes(max_latency_seconds=2.41, max_size_mb=4.31),
        "static_schema": BenchmarkOutcomes(max_latency_seconds=13.91, max_size_mb=4.57),
    },
    "r500_m50": {
        "dynamic_schema": BenchmarkOutcomes(max_latency_seconds=0.45, max_size_mb=0.32),
        "normalized_narrow": BenchmarkOutcomes(max_latency_seconds=1.42, max_size_mb=2.12),
        "static_schema": BenchmarkOutcomes(max_latency_seconds=6.94, max_size_mb=2.29),
    },
    "r1000_m50": {
        "dynamic_schema": BenchmarkOutcomes(max_latency_seconds=0.82, max_size_mb=0.58),
        "normalized_narrow": BenchmarkOutcomes(max_latency_seconds=2.25, max_size_mb=3.96),
        "static_schema": BenchmarkOutcomes(max_latency_seconds=13.90, max_size_mb=4.35),
    },
    "r2000_m50": {
        "dynamic_schema": BenchmarkOutcomes(max_latency_seconds=2.15, max_size_mb=1.16),
        "normalized_narrow": BenchmarkOutcomes(max_latency_seconds=4.38, max_size_mb=7.87),
        "static_schema": BenchmarkOutcomes(max_latency_seconds=27.55, max_size_mb=8.67),
    },
}


def _static_columns() -> list[SQLColumn]:
    return [
        SQLColumn(name="timestamp", type="TIMESTAMP WITH TIME ZONE"),
        SQLColumn(name="metric", type="TEXT"),
        SQLColumn(name="value", type="DOUBLE PRECISION", nullable=True),
    ]


def _unique_suffix(request: pytest.FixtureRequest) -> str:
    digest = hashlib.sha1(request.node.nodeid.encode("utf-8"), usedforsecurity=False).hexdigest()
    return digest[:12]


def _generate_dynamic_rows(num_rows: int, num_metrics: int) -> list[dict[str, Any]]:
    base_time = datetime.now(UTC)
    rng = random.Random(_scenario_seed(namespace="dynamic", num_rows=num_rows, num_metrics=num_metrics))
    rows: list[dict[str, Any]] = []
    for index in range(num_rows):
        row: dict[str, Any] = {
            "time": base_time + timedelta(seconds=index),
        }
        for metric_index in range(num_metrics):
            row[f"metric_{metric_index + 1}"] = rng.uniform(-1_000.0, 1_000.0)
        rows.append(row)
    return rows


def _generate_narrow_rows(num_rows: int, num_metrics: int) -> list[dict[str, Any]]:
    base_time = datetime.now(UTC)
    rng = random.Random(_scenario_seed(namespace="narrow", num_rows=num_rows, num_metrics=num_metrics))
    rows: list[dict[str, Any]] = []
    for index in range(num_rows):
        current_time = base_time + timedelta(seconds=index)
        rows.extend(
            {
                "timestamp": current_time,
                "metric": f"metric-{metric_index + 1}",
                "value": rng.uniform(-1_000.0, 1_000.0),
            }
            for metric_index in range(num_metrics)
        )
    return rows


def _dynamic_row_count(engine: Engine, writer: SqlWriter) -> int:
    assert isinstance(writer, DynamicSchemaSqlWriter)
    return table_count(engine, writer.table_name, schema=writer.schema)


def _normalized_row_count(engine: Engine, writer: SqlWriter) -> int:
    assert isinstance(writer, NormalizedNarrowSqlWriter)
    return table_count(engine, writer.data_table_name, schema=writer.schema)


def _dynamic_table_refs(writer: SqlWriter) -> Sequence[tuple[str | None, str]]:
    assert isinstance(writer, DynamicSchemaSqlWriter)
    return ((writer.schema, writer.table_name),)


def _normalized_table_refs(writer: SqlWriter) -> Sequence[tuple[str | None, str]]:
    assert isinstance(writer, NormalizedNarrowSqlWriter)
    return (
        (writer.schema, writer.data_table_name),
        (writer.schema, writer.lookup_table_name),
    )


def _static_row_count(engine: Engine, writer: SqlWriter) -> int:
    assert isinstance(writer, StaticSchemaSqlWriter)
    return table_count(engine, writer.table_name, schema=writer.schema)


def _static_table_refs(writer: SqlWriter) -> Sequence[tuple[str | None, str]]:
    assert isinstance(writer, StaticSchemaSqlWriter)
    return ((writer.schema, writer.table_name),)


WRITER_CONFIGS: Sequence[WriterBenchmarkConfig] = (
    WriterBenchmarkConfig(
        name="dynamic_schema",
        create_writer=lambda engine, suffix: DynamicSchemaSqlWriter(
            engine=engine,
            table_name=f"bench_dynamic_{suffix}",
            table_creation_factory=lambda schema, table, columns: create_tsdb_table_query(
                schema=schema,
                table=table,
                columns=columns,
                partition_column=None,
                index_columns=[],
            ),
            schema="public",
            default_column_type="DOUBLE PRECISION",
        ),
        generate_rows=_generate_dynamic_rows,
        row_count_fn=_dynamic_row_count,
        table_refs_fn=_dynamic_table_refs,
    ),
    WriterBenchmarkConfig(
        name="normalized_narrow",
        create_writer=lambda engine, suffix: NormalizedNarrowSqlWriter(
            engine=engine,
            data_table_name=f"bench_narrow_data_{suffix}",
            lookup_table_name=f"bench_narrow_lookup_{suffix}",
            schema="public",
        ),
        generate_rows=_generate_narrow_rows,
        row_count_fn=_normalized_row_count,
        table_refs_fn=_normalized_table_refs,
    ),
    WriterBenchmarkConfig(
        name="static_schema",
        create_writer=lambda engine, suffix: StaticSchemaSqlWriter(
            engine=engine,
            table_name=f"bench_static_{suffix}",
            columns=_static_columns(),
            table_creation_factory=lambda schema, table, columns: create_tsdb_table_query(
                schema=schema,
                table=table,
                columns=columns,
                partition_column=None,
                index_columns=['metric'],
                time_column="timestamp",
            ),
            schema="public",
        ),
        generate_rows=_generate_narrow_rows,
        row_count_fn=_static_row_count,
        table_refs_fn=_static_table_refs,
    ),
)
@pytest.mark.parametrize("writer_config", WRITER_CONFIGS, ids=lambda cfg: cfg.name)
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda sc: sc.name)
def test_sql_writer_benchmarks(
    writer_config: WriterBenchmarkConfig,
    scenario: BenchmarkScenario,
    tsdb_engine: Engine,
    request: pytest.FixtureRequest,
    record_property: Callable[[str, Any], None],
):
    scenario_expectations = EXPECTED_OUTCOMES[scenario.name]
    expected = scenario_expectations.get(writer_config.name)
    if expected is None:
        pytest.skip(
            f"No expectations configured for writer '{writer_config.name}' in scenario '{scenario.name}'",
        )

    suffix = _unique_suffix(request)
    writer = writer_config.create_writer(tsdb_engine, suffix)
    rows = writer_config.generate_rows(scenario.num_rows, scenario.num_metrics)

    total_rows = len(rows)

    start = time.perf_counter()
    writer.write_many(rows)
    duration = time.perf_counter() - start

    row_count = writer_config.row_count_fn(tsdb_engine, writer)
    assert row_count == total_rows, (
        f"Expected {total_rows} rows, found {row_count} in database for {writer_config.name}"
    )

    rows_per_second = total_rows / duration if duration > 0 else float("inf")
    record_property("duration_seconds", round(duration, 6))
    record_property("rows_per_second", round(rows_per_second, 2))
    print(
        f"[BENCHMARK] scenario={scenario.name} writer={writer_config.name} "
        f"duration_seconds={duration:.6f} rows_per_second={rows_per_second:.2f}",
    )
    assert duration <= expected.max_latency_seconds, (
        f"Ingestion took {duration:.2f}s exceeding threshold {expected.max_latency_seconds:.2f}s"
    )

    total_size_mb = sum(
        table_size_mb(tsdb_engine, table, schema=schema)
        for schema, table in writer_config.table_refs_fn(writer)
    )
    record_property("database_size_mb", round(total_size_mb, 2))
    print(
        f"[BENCHMARK] scenario={scenario.name} writer={writer_config.name} "
        f"database_size_mb={total_size_mb:.2f}",
    )
    _append_benchmark_result(
        scenario=scenario.name,
        writer=writer_config.name,
        rows=total_rows,
        duration=duration,
        size_mb=total_size_mb,
    )
    assert total_size_mb <= expected.max_size_mb, (
        f"Database size {total_size_mb:.2f}MB exceeded threshold {expected.max_size_mb:.2f}MB"
    )

    writer.close()
