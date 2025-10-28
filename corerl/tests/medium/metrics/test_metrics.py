import datetime as dt
from copy import deepcopy
from typing import NamedTuple

import pandas as pd
import pytest
import pytz
from sqlalchemy import Engine

from corerl.eval.metrics.base import MetricsWriterProtocol
from tests.infrastructure.utils import get_fixture

MetricsTableFixture = tuple[MetricsWriterProtocol, dt.datetime, dt.timedelta]


@pytest.fixture()
def populated_wide_metrics_table(metrics_table: MetricsWriterProtocol):
    """Populate metrics table in wide format (default) - 5 steps with 2 metrics each."""
    start_time = dt.datetime(2023, 7, 13, 6, tzinfo=pytz.UTC)
    curr_time = deepcopy(start_time)
    delta = dt.timedelta(hours=1)
    for i in range(5):
        metrics_table.write(
            agent_step=i,
            metric="q",
            value=i,
            timestamp=curr_time.isoformat(),
        )
        metrics_table.write(
            agent_step=i,
            metric="reward",
            value=2*i,
            timestamp=curr_time.isoformat(),
        )
        curr_time += delta
    metrics_table.flush()
    return metrics_table, start_time, delta

@pytest.mark.parametrize(
    "fixture_name, expected_len",
    [
        ("populated_wide_metrics_table", 5),  # 5 steps with 2 metrics per step
        ("populated_wide_metrics_table_sparse", 7),  # 5 steps with 2 metrics, plus 2 steps with 1 metric
    ],
)
def test_metrics_read_by_time(
    tsdb_engine: Engine,
    request: pytest.FixtureRequest,
    fixture_name: str,
    expected_len: int,
):
    metrics_table, start_time, delta = get_fixture(request, fixture_name, MetricsTableFixture)
    table_name = metrics_table.table_name

    with tsdb_engine.connect() as conn:
        metrics_df = pd.read_sql_table(table_name, con=conn)
    assert len(metrics_df) == expected_len

    start_ind = 1
    end_ind = 3
    query_start = start_time + (start_ind * delta)
    query_end = start_time + (end_ind * delta)
    rewards_df = metrics_table.read("reward", start_time=query_start, end_time=query_end)
    q_df = metrics_table.read("q", start_time=query_start, end_time=query_end)

    expected_times = [pd.Timestamp(start_time + (i * delta)) for i in range(start_ind, end_ind + 1)]
    expected_rewards_df = pd.DataFrame({
        "time": expected_times,
        "reward": [float(2 * i) for i in range(start_ind, end_ind + 1)],
    })
    expected_q_df = pd.DataFrame({
        "time": expected_times,
        "q": [float(i) for i in range(start_ind, end_ind + 1)],
    })
    pd.testing.assert_frame_equal(rewards_df.reset_index(drop=True), expected_rewards_df)
    pd.testing.assert_frame_equal(q_df.reset_index(drop=True), expected_q_df)



class MetricsReadByStepCase(NamedTuple):
    fixture_name: str
    start_step: int | None
    end_step: int
    agent_step: list[int]
    reward: list[float]
    q: list[float]

@pytest.mark.parametrize(
    "case",
    [
        MetricsReadByStepCase(
            fixture_name="populated_wide_metrics_table",
            start_step=None,
            end_step=3,
            agent_step=[0, 1, 2, 3],
            reward=[0.0, 2.0, 4.0, 6.0],
            q=[0.0, 1.0, 2.0, 3.0],
        ),
        MetricsReadByStepCase(
            fixture_name="populated_wide_metrics_table_sparse",
            start_step=None,
            end_step=3,
            agent_step=[0, 1, 2, 3],
            reward=[0.0, 2.0, 4.0, 6.0],
            q=[0.0, 1.0, 2.0, 3.0],
        ),
    ],
)
def test_metrics_read_by_step(request: pytest.FixtureRequest, case: MetricsReadByStepCase):
    metrics_table, *_ = get_fixture(request, case.fixture_name, MetricsTableFixture)

    rewards_df = metrics_table.read("reward", step_start=case.start_step, step_end=case.end_step)
    q_df = metrics_table.read("q", step_start=case.start_step, step_end=case.end_step)

    expected_reward_df = pd.DataFrame({
        "agent_step": case.agent_step,
        "reward": case.reward,
    })
    expected_q_df = pd.DataFrame({
        "agent_step": case.agent_step,
        "q": case.q,
    })

    pd.testing.assert_frame_equal(rewards_df.reset_index(drop=True), expected_reward_df)
    pd.testing.assert_frame_equal(q_df.reset_index(drop=True), expected_q_df)



class MetricsReadByMetricCase(NamedTuple):
    fixture_name: str
    time: list[pd.Timestamp]
    agent_step: list[int]
    reward: list[float]
    q: list[float]

@pytest.mark.parametrize(
    "case",
    [
        MetricsReadByMetricCase(
            fixture_name="populated_wide_metrics_table",
            time=[
                pd.Timestamp(dt.datetime(2023, 7, 13, 6, tzinfo=pytz.UTC)),
                pd.Timestamp(dt.datetime(2023, 7, 13, 7, tzinfo=pytz.UTC)),
                pd.Timestamp(dt.datetime(2023, 7, 13, 8, tzinfo=pytz.UTC)),
                pd.Timestamp(dt.datetime(2023, 7, 13, 9, tzinfo=pytz.UTC)),
                pd.Timestamp(dt.datetime(2023, 7, 13, 10, tzinfo=pytz.UTC)),
            ],
            agent_step=list(range(5)),
            reward=[2.0 * i for i in range(5)],
            q=[float(i) for i in range(5)],
        ),
        MetricsReadByMetricCase(
            fixture_name="populated_wide_metrics_table_sparse",
            time=[
                pd.Timestamp(dt.datetime(2023, 7, 13, 6, tzinfo=pytz.UTC)),
                pd.Timestamp(dt.datetime(2023, 7, 13, 7, tzinfo=pytz.UTC)),
                pd.Timestamp(dt.datetime(2023, 7, 13, 8, tzinfo=pytz.UTC)),
                pd.Timestamp(dt.datetime(2023, 7, 13, 9, tzinfo=pytz.UTC)),
                pd.Timestamp(dt.datetime(2023, 7, 13, 10, tzinfo=pytz.UTC)),
                pd.Timestamp(dt.datetime(2023, 7, 13, 11, tzinfo=pytz.UTC)),
                pd.Timestamp(dt.datetime(2023, 7, 13, 11, tzinfo=pytz.UTC)),
            ],
            agent_step=[0, 1, 2, 3, 4, 5, 6],
            reward=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            q=[0.0, 1.0, 2.0, 3.0, 4.0],
        ),
    ],
)
def test_metrics_read_by_metric(request: pytest.FixtureRequest, case: MetricsReadByMetricCase):
    metrics_table, *_ = get_fixture(request, case.fixture_name, MetricsTableFixture)

    rewards_df = metrics_table.read("reward")
    q_df = metrics_table.read("q")

    expected_reward_df = pd.DataFrame({
        "time": case.time,
        "agent_step": case.agent_step,
        "reward": case.reward,
    })

    # For wide format, q only has values for first 5 steps
    q_time = case.time[:len(case.q)]
    q_agent_step = case.agent_step[:len(case.q)]
    expected_q_df = pd.DataFrame({
        "time": q_time,
        "agent_step": q_agent_step,
        "q": case.q,
    })

    pd.testing.assert_frame_equal(rewards_df.reset_index(drop=True), expected_reward_df)
    pd.testing.assert_frame_equal(q_df.reset_index(drop=True), expected_q_df)


@pytest.mark.parametrize(
    "fixture_name",
    ["metrics_table", "wide_metrics_table"],
)
def test_disconnect_between_writes(
    request: pytest.FixtureRequest,
    tsdb_engine: Engine,
    fixture_name: str,
):
    metrics_table: MetricsWriterProtocol = request.getfixturevalue(fixture_name)

    # Write first batch of metrics
    metrics_table.write(agent_step=0, metric="q", value=1.0, timestamp="2023-07-13T06:00:00+00:00")
    metrics_table.write(agent_step=0, metric="reward", value=2.0, timestamp="2023-07-13T06:00:00+00:00")
    # Simulate disconnect
    tsdb_engine.dispose()

    # Write second batch of metrics
    metrics_table.write(agent_step=1, metric="q", value=3.0, timestamp="2023-07-13T07:00:00+00:00")
    metrics_table.write(agent_step=1, metric="reward", value=4.0, timestamp="2023-07-13T07:00:00+00:00")
    rewards_df = metrics_table.read("reward")
    q_df = metrics_table.read("q")

    expected_rewards_df = pd.DataFrame({
        "time": [
            pd.Timestamp("2023-07-13T06:00:00+00:00"),
            pd.Timestamp("2023-07-13T07:00:00+00:00"),
        ],
        "agent_step": [0, 1],
        "reward": [2.0, 4.0],
    })

    expected_q_df = pd.DataFrame({
        "time": [
            pd.Timestamp("2023-07-13T06:00:00+00:00"),
            pd.Timestamp("2023-07-13T07:00:00+00:00"),
        ],
        "agent_step": [0, 1],
        "q": [1.0, 3.0],
    })

    pd.testing.assert_frame_equal(rewards_df.reset_index(drop=True), expected_rewards_df)
    pd.testing.assert_frame_equal(q_df.reset_index(drop=True), expected_q_df)


# ---------------------------------------------------------------------------- #
#                                  Wide Tests                                  #
# ---------------------------------------------------------------------------- #

@pytest.fixture()
def populated_wide_metrics_table_sparse(wide_metrics_table: MetricsWriterProtocol):
    """Populate wide metrics table with sparse data - some steps have only partial metrics."""
    start_time = dt.datetime(2023, 7, 13, 6, tzinfo=pytz.UTC)
    curr_time = deepcopy(start_time)
    delta = dt.timedelta(hours=1)
    for i in range(5):
        wide_metrics_table.write(
            agent_step=i,
            metric="q",
            value=i,
            timestamp=curr_time.isoformat(),
        )
        wide_metrics_table.write(
            agent_step=i,
            metric="reward",
            value=2 * i,
            timestamp=curr_time.isoformat(),
        )
        curr_time += delta
    # add two reward values without associated q (tests sparse data handling)
    wide_metrics_table.write(
        agent_step=5,
        metric="reward",
        value=2 * 5,
        timestamp=curr_time.isoformat(),
    )
    wide_metrics_table.write(
        agent_step=6,
        metric="reward",
        value=2 * 6,
        timestamp=curr_time.isoformat(),
    )
    wide_metrics_table.flush()
    return wide_metrics_table, start_time, delta


def test_db_metrics_write_wide_sparse(
    tsdb_engine: Engine,
    populated_wide_metrics_table_sparse: MetricsTableFixture,
):
    """Test that wide format correctly handles sparse data (missing metrics for some steps)."""
    metrics_table, start_time, delta = populated_wide_metrics_table_sparse

    with tsdb_engine.connect() as conn:
        metrics_df = pd.read_sql_table(metrics_table.table_name, con=conn)

    assert len(metrics_df) == 7

    # Create expected dataframe for wide format with step-based flush semantics
    agent_steps = [0, 1, 2, 3, 4, 5, 6]
    times = [
        pd.Timestamp(start_time),
        pd.Timestamp(start_time + delta),
        pd.Timestamp(start_time + (2 * delta)),
        pd.Timestamp(start_time + (3 * delta)),
        pd.Timestamp(start_time + (4 * delta)),
        pd.Timestamp(start_time + (5 * delta)),  # step 5
        pd.Timestamp(start_time + (5 * delta)),  # step 6 same timestamp, separate row
    ]
    q_values = [0.0, 1.0, 2.0, 3.0, 4.0, None, None]
    reward_values = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    expected_df = pd.DataFrame({
        'time': times,
        'agent_step': agent_steps,
        'q': q_values,
        'reward': reward_values,
    })

    pd.testing.assert_frame_equal(
        metrics_df.reindex(columns=['time', 'agent_step', 'q', 'reward']),
        expected_df,
    )


# ---------------------------------------------------------------------------- #
#                              Prefix Reading Tests                           #
# ---------------------------------------------------------------------------- #

@pytest.fixture()
def populated_wide_metrics_table_with_prefixes(metrics_table: MetricsWriterProtocol):
    """Create wide metrics table (default) with various metrics that have common prefixes."""
    start_time = dt.datetime(2023, 7, 13, 6, tzinfo=pytz.UTC)
    curr_time = deepcopy(start_time)
    delta = dt.timedelta(hours=1)

    for i in range(3):
        # Training metrics
        metrics_table.write(
            agent_step=i,
            metric="train_loss",
            value=i,
            timestamp=curr_time.isoformat(),
        )
        metrics_table.write(
            agent_step=i,
            metric="train_accuracy",
            value=10 + i,
            timestamp=curr_time.isoformat(),
        )

        # Validation metrics
        metrics_table.write(
            agent_step=i,
            metric="val_loss",
            value=2 * i,
            timestamp=curr_time.isoformat(),
        )
        metrics_table.write(
            agent_step=i,
            metric="val_accuracy",
            value=20 + i,
            timestamp=curr_time.isoformat(),
        )

        # Test metrics (only for first 2 steps)
        if i < 2:
            metrics_table.write(
                agent_step=i,
                metric="test_loss",
                value=3 * i,
                timestamp=curr_time.isoformat(),
            )

        curr_time += delta

    metrics_table.flush()
    return metrics_table, start_time, delta


def test_metrics_read_by_prefix_all_metrics(
    populated_wide_metrics_table_with_prefixes: MetricsTableFixture,
):
    """Test reading all metrics by prefix (full dataset)."""
    metrics_table, *_ = populated_wide_metrics_table_with_prefixes

    # Read all training metrics
    train_df = metrics_table.read("train_", prefix_match=True)

    # Should contain train_loss and train_accuracy columns
    expected_columns = {"time", "agent_step", "train_loss", "train_accuracy"}
    assert set(train_df.columns) == expected_columns

    # Should have 3 rows (for agent steps 0, 1, 2)
    assert len(train_df) == 3

    # Check specific values
    assert train_df.loc[0, "train_loss"] == 0.0
    assert train_df.loc[1, "train_loss"] == 1.0
    assert train_df.loc[2, "train_loss"] == 2.0

    assert train_df.loc[0, "train_accuracy"] == 10.0
    assert train_df.loc[1, "train_accuracy"] == 11.0
    assert train_df.loc[2, "train_accuracy"] == 12.0


def test_metrics_read_by_prefix_with_missing_data(
    populated_wide_metrics_table_with_prefixes: MetricsTableFixture,
):
    """Test reading metrics by prefix where some metrics have missing data."""
    metrics_table, *_ = populated_wide_metrics_table_with_prefixes

    # Read all test metrics (only exists for steps 0 and 1)
    test_df = metrics_table.read("test_", prefix_match=True)

    expected_columns = {"time", "agent_step", "test_loss"}
    assert set(test_df.columns) == expected_columns

    # Should have 2 rows (for agent steps 0, 1)
    assert len(test_df) == 2

    # Check values
    assert test_df.loc[0, "test_loss"] == 0.0
    assert test_df.loc[1, "test_loss"] == 3.0


def test_metrics_read_by_prefix_with_missing_data_wide(
    populated_wide_metrics_table_with_prefixes: MetricsTableFixture,
):
    """Test reading metrics by prefix where some metrics have missing data (wide format)."""
    metrics_table, *_ = populated_wide_metrics_table_with_prefixes

    # Read all test metrics (only exists for steps 0 and 1)
    test_df = metrics_table.read("test_", prefix_match=True)

    expected_columns = {"time", "agent_step", "test_loss"}
    assert set(test_df.columns) == expected_columns

    # Should have 2 rows (for agent steps 0, 1)
    assert len(test_df) == 2

    # Check values
    assert test_df.loc[0, "test_loss"] == 0.0
    assert test_df.loc[1, "test_loss"] == 3.0


def test_metrics_read_by_prefix_step_filtering(
    populated_wide_metrics_table_with_prefixes: MetricsTableFixture,
):
    """Test reading metrics by prefix with step filtering."""
    metrics_table, *_ = populated_wide_metrics_table_with_prefixes

    # Read validation metrics for steps 1-2 only
    val_df = metrics_table.read("val_", prefix_match=True, step_start=1, step_end=2)

    expected_columns = {"agent_step", "val_loss", "val_accuracy"}
    assert set(val_df.columns) == expected_columns

    # Should have 2 rows (for agent steps 1, 2)
    assert len(val_df) == 2

    # Check values
    assert val_df.loc[0, "agent_step"] == 1
    assert val_df.loc[1, "agent_step"] == 2
    assert val_df.loc[0, "val_loss"] == 2.0
    assert val_df.loc[1, "val_loss"] == 4.0


def test_metrics_read_by_prefix_time_filtering(
    populated_wide_metrics_table_with_prefixes: MetricsTableFixture,
):
    """Test reading metrics by prefix with time filtering."""
    metrics_table, start_time, delta = populated_wide_metrics_table_with_prefixes

    # Read training metrics for middle time period only
    query_start = start_time + delta  # Second hour
    query_end = start_time + (2 * delta)  # Third hour

    train_df = metrics_table.read("train_", prefix_match=True, start_time=query_start, end_time=query_end)

    expected_columns = {"time", "train_loss", "train_accuracy"}
    assert set(train_df.columns) == expected_columns

    # Should have 2 rows (for hours 1 and 2)
    assert len(train_df) == 2


def test_metrics_read_by_prefix_no_matches(
    populated_wide_metrics_table_with_prefixes: MetricsTableFixture,
):
    """Test reading metrics by prefix when no metrics match."""
    metrics_table, *_ = populated_wide_metrics_table_with_prefixes

    # Try to read metrics with a prefix that doesn't exist
    nonexistent_df = metrics_table.read("nonexistent_", prefix_match=True)

    # Should return empty dataframe with expected base columns
    expected_columns = {"time", "agent_step"}
    assert set(nonexistent_df.columns) == expected_columns
    assert len(nonexistent_df) == 0
