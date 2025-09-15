import logging
import shutil
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from corerl.data_pipeline.datatypes import StageCode, Transition
from corerl.environment.reward.config import Optimization
from corerl.eval.raw_data import raw_data_eval_for_tag
from corerl.state import AppState
from tabulate import tabulate

from coreoffline.config import ReportConfig
from coreoffline.data_analysis.plotting import (
    plot_chunk_histogram,
    plot_nan_histogram,
    plot_sensor_data,
    plot_sensor_histogram,
)

if TYPE_CHECKING:
    pass


log = logging.getLogger(__name__)


def get_tags(data: list[pd.DataFrame]) -> list[str]:
    return list(data[0].columns)


def make_stat_table(
        cfg: ReportConfig,
        data: list[pd.DataFrame],
        stages: list[StageCode],
        output_path: Path,
    ) -> None:

    if not cfg.stat_table_enabled:
        return

    tags = get_tags(data)
    table_data: list[list[str]] = []
    headers = ['tag']
    for tag in tags:
        if tag is cfg.tags_to_exclude:
            continue
        for stage_i, df in enumerate(data):
            row = [f"{tag} - {stages[stage_i].name}"]
            headers = ['tag']
            tag_stats = raw_data_eval_for_tag(df, tag)
            for key, value in tag_stats.items():
                headers.append(key)
                row.append(str(round(value, 2)))
            table_data.append(row)
    table_data.insert(0, headers)
    table_str = tabulate(table_data, headers='firstrow', tablefmt='grid')
    (output_path / 'sensor_report.txt').write_text(table_str, encoding='utf-8')


def make_distribution_plots(
        cfg: ReportConfig,
        data: list[pd.DataFrame],
        stages: list[StageCode],
        output_path: Path,
    ) -> None:

    if not cfg.hist_enabled:
        return

    tags = get_tags(data)

    show_hist_mean = cfg.hist_show_mean
    percentiles = cfg.hist_percentiles
    num_bins = cfg.hist_num_bins

    log.info('Generating Distribution Plots...')
    for tag in tags:
        for stage_i, df in enumerate(data):
            stage_name = stages[stage_i].name
            tag_stage_output_path = output_path / tag / stage_name
            tag_stage_output_path.mkdir(parents=True, exist_ok=True)
            plot_sensor_data(
                df,
                tag,
                save_path=tag_stage_output_path / f'{tag}_sensor_data_{stage_name}.png',
                title=f'{tag} Sensor Data - {stage_name}',
            )
            plot_sensor_histogram(
                df,
                tag,
                save_path=tag_stage_output_path / f'{tag}_sensor_histogram_{stage_name}.png',
                title=f'{tag} Histogram - {stage_name}',
                show_mean=show_hist_mean,
                percentiles=percentiles,
                bins=num_bins,
            )
            plot_nan_histogram(
                df,
                tag,
                save_path=tag_stage_output_path / f'{tag}_nan_histogram_{stage_name}.png',
                title=f'{tag} NaN Histogram - {stage_name}',
                show_mean=show_hist_mean,
                percentiles=percentiles,
                bins=num_bins,
            )
            plot_chunk_histogram(
                df,
                tag,
                save_path=tag_stage_output_path / f'{tag}_chunk_histogram_{stage_name}.png',
                title=f'{tag} Chunk Histogram - {stage_name}',
                show_mean=show_hist_mean,
                percentiles=percentiles,
                bins=num_bins,
            )


def get_tag_pairs(
        cfg: ReportConfig,
        data: list[pd.DataFrame],
    ) -> list[tuple[str, str]]:
    tag_info = cfg.cross_corr_tags
    if tag_info is None:  # all pairs of tags
        tags = get_tags(data)
        pairs = list(combinations(tags, 2))
    elif len(tag_info) == 0:
        return []
    elif isinstance(tag_info[0], str):  # all pairs of tags specified in cfg.cross_corr_tags
        str_tags = tag_info
        pairs = list(combinations(str_tags, 2))
    else:  # only pairs of tags specified in cfg.cross_corr_tags
        for pair in tag_info:
            assert len(pair) == 2
        pairs = tag_info

    pairs = [tuple(pair) for pair in pairs]
    return cast(list[tuple[str, str]], pairs)


def make_cross_correlation_table(
        cfg: ReportConfig,
        data: list[pd.DataFrame],
        stages: list[StageCode],
        output_path: Path,
        ) -> None:

    if not cfg.cross_corr_enabled:
        return

    log.info("Making cross correlation table")
    tag_pairs = get_tag_pairs(cfg, data)
    all_tags = get_tags(data)
    max_lag = cfg.cross_corr_max_lag
    table = [['Stage', 'tag 1', 'tag 2', 'Max Cross Correlation', 'Lag for Max Cross Correlation']]
    for stage_i, stage in enumerate(stages):
        df = data[stage_i]
        for tag_pair in tag_pairs:
            tag_1, tag_2 = tag_pair

            assert tag_1 in all_tags
            assert tag_2 in all_tags
            cc, lag, _ = cross_correlation(df, tag_1, tag_2, max_lag)
            row = [stage.name, tag_1, tag_2, cc, lag]
            table.append(row)  # type: ignore

    table_str = tabulate(table, headers='firstrow', tablefmt='grid')
    (output_path / 'cross_correlation.txt').write_text(table_str, encoding='utf-8')


def standardize(x: np.ndarray, mask: np.ndarray):
    """
    Standardization that ignores all values not included in the mask.
    Masked values are output as 0.
    """
    if np.all(~mask):
        return np.zeros_like(x)
    std = np.std(x[mask])
    if std == 0:
        return np.zeros_like(x)
    x = x.copy()
    mean = np.mean(x[mask])
    return mask * ((x - mean) / std)


def correlate(
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int,
    ) -> np.ndarray:
    """
    Unnormalized cross correlation for lags in [-max_lag, +max_lag]
    """
    assert (not np.any(np.isnan(x))) and (not np.any(np.isnan(y)))
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x_, y_ = x[:lag], y[-lag:]  # note: lag is negative, so we are implicity flipping the sign for indexing
        elif lag > 0:
            x_, y_ = x[lag:], y[:-lag]
        else:
            x_, y_ = x, y
        corr = np.correlate(x_, y_)
        assert len(corr) == 1
        corr = corr[0]
        correlations.append(corr)

    return np.array(correlations)


def cross_correlation(
        df: pd.DataFrame,
        tag_1: str,
        tag_2: str,
        max_lag: int,
    ) -> tuple[float, float, np.ndarray]:
    """
    Computes cross correlation between tag_1 and tag_2, where the lag may vary between [-max_lag, +max_lag].

    Note that the returned variables can be greater than one.
    """

    x = df[tag_1].to_numpy()
    y = df[tag_2].to_numpy()

    x_mask = ~np.isnan(x)  # where x is not nan
    y_mask = ~np.isnan(y)  # where y is not nan

    if np.all(~x_mask) or np.all(~y_mask):  # either x or y is all nan
        return -np.inf, 0, np.array([-np.inf])

    # zero-out nans
    x = np.where(np.isnan(x), 0, x)
    y = np.where(np.isnan(y), 0, y)

    # Standardize x and y, but only using statistics from non-nan entries.
    # Note that standardization is performed on the masked arrays,
    # not the subset of data that is passed to correlate.
    x, y = standardize(x, x_mask), standardize(y, y_mask)

    # Compute unnormalized cross correlation between x and y.
    # Since previous nans are now 0, they do not affect unnormalized result
    corr = correlate(x, y, max_lag)

    # for each lag, we can find the number of non-nan entries in the sum
    # by taking the cross correlation of x_mask and y_mask
    num_non_nan = correlate(x_mask.astype(int), y_mask.astype(int), max_lag)

    # make zeros nan, to signal when cross corr not valid
    num_non_nan = np.where(num_non_nan == 0, np.nan, num_non_nan)

    cross_corr = corr / num_non_nan  # normalize cross corr
    cross_corr = np.where(np.isnan(cross_corr), -np.inf, cross_corr)
    max_idx = np.argmax(cross_corr)
    lags = np.arange(-max_lag, max_lag + 1)
    return cross_corr[max_idx], lags[max_idx], cross_corr


def calculate_contiguous_sequence_lengths(
        transitions: list[Transition],
        time_threshold: timedelta,
    ):
    """
    Calculate lengths of contiguous transition sequences.
    Transitions are considered contiguous if the time gap between consecutive transitions
    is less than the specified threshold.
    """
    if not transitions:
        return []

    sequence_lengths = []
    current_length = 1

    for i in range(1, len(transitions)):
        prev_transition = transitions[i - 1]
        curr_transition = transitions[i]

        # Check if transitions are contiguous based on timestamp
        if are_transitions_contiguous(prev_transition, curr_transition, time_threshold):
            current_length += 1
        else:
            sequence_lengths.append(current_length)
            current_length = 1

    # Add the last sequence
    sequence_lengths.append(current_length)

    return sequence_lengths


def are_transitions_contiguous(
    prev_transition: Transition,
    curr_transition: Transition,
    time_threshold: timedelta,
):
    """
    Determine if two transitions are contiguous based on timestamp proximity.
    """
    # Check if both transitions have timestamps
    prev_timestamp = prev_transition.end_time
    curr_timestamp = curr_transition.start_time

    if prev_timestamp is None or curr_timestamp is None:
        return False

    # Calculate time difference
    time_diff = curr_timestamp - prev_timestamp
    return time_diff <= time_threshold


def get_sequence_stats(cfg: ReportConfig, sequence_lengths: list[int]):
    return_dict = {
        'Total Sequences': len(sequence_lengths),
        'Average Sequence Length': np.mean(sequence_lengths),
        'Min Sequence Length': np.min(sequence_lengths),
        'Max Sequence Length': np.max(sequence_lengths),
    }
    # Add percentiles
    for p in cfg.transition_percentiles:
        percentile_value = np.percentile(sequence_lengths, p * 100)
        return_dict[f'P{int(p * 100)} Sequence Length'] = percentile_value
    return return_dict


def calculate_violation_periods(
        satisfaction_df: pd.DataFrame,
        metric_name: str,
        obs_period: timedelta,
    ) -> list[timedelta]:
    """
    Calculate consecutive violation periods from satisfaction data.
    Returns list of violation period durations.
    """
    if satisfaction_df.empty:
        return []

    # Ensure dataframe is sorted by timestamp
    satisfaction_df = satisfaction_df.sort_index()

    # Find violations (satisfaction < 1)
    violations = satisfaction_df[metric_name] < 1

    if not violations.any():
        return []

    # Find start and end of consecutive violation periods
    violation_periods: list[int] = []
    in_violation = False
    violation_start_idx: int | None = None

    for idx, is_violation in violations.items():
        assert isinstance(idx, int)
        if is_violation and not in_violation:
            # Start of violation period
            violation_start_idx = idx
            in_violation = True
        elif not is_violation and in_violation:
            # End of violation period
            assert violation_start_idx is not None
            period_duration = idx - violation_start_idx
            violation_periods.append(period_duration)
            in_violation = False
            violation_start_idx = None

    # Handle case where violation period extends to end of data
    if in_violation and violation_start_idx is not None:
        last_idx = satisfaction_df.index[-1]
        assert isinstance(last_idx, int)
        period_duration = last_idx - violation_start_idx
        violation_periods.append(period_duration)

    return [vp * obs_period for vp in violation_periods]


def get_violation_period_stats(violation_periods: list[timedelta], percentiles: list[float]):
    """
    Calculate statistics for violation periods.
    """
    if not violation_periods:
        return {'No violation periods found': ''}

    # Convert timedeltas to total seconds for calculations
    period_seconds = [period.total_seconds() / 60 for period in violation_periods]

    stats = {
        'Total Violation Periods': str(len(violation_periods)),
        'Min Violation Period (minutes)': f'{min(period_seconds):.2f}',
        'Max Violation Period (minutes)': f'{max(period_seconds):.2f}',
        'Mean Violation Period (minutes)': f'{np.mean(period_seconds):.2f}',
    }

    # Add percentiles
    for p in percentiles:
        percentile_value = np.percentile(period_seconds, p * 100)
        stats[f'P{int(p * 100)} Violation Period (minutes)'] = f'{percentile_value:.2f}'

    return stats


def make_goal_violations_table(
        cfg: ReportConfig,
        output_path: Path,
        app_state: AppState,
        start_time: datetime,
        end_time: datetime,
    ):
    """
    Generate goal violations table and save to file.
    """

    if app_state.cfg.pipeline.reward is None:
        return

    log.info("Generating goal violations statistics...")

    # Prepare table data
    table_data: list[list[str]] = [
        ['start time of report gen.', str(start_time)],
        ['end time of report gen.', str(end_time)],
    ]

    headers = [
        'Metric',
        'Value',
    ]

    assert app_state.cfg.pipeline.reward is not None
    priorities = app_state.cfg.pipeline.reward.priorities

    for priority_idx, priority in enumerate(priorities):
        # Skip optimization priorities - they use a different metric
        if isinstance(priority, Optimization):
            continue

        metric_name = f'priority_{priority_idx}_satisfaction'  # from goals.py

        satisfaction_df = app_state.metrics.read(
            metric=metric_name,
            start_time=start_time,
            end_time=end_time,
        )

        if not satisfaction_df.empty:
            # Count violations (entries where satisfaction < 1)
            violations = (satisfaction_df[metric_name] < 1).sum()
            table_data.append([f'Priority {priority_idx} Violations', str(violations)])

            # Calculate satisfaction rate
            total_entries = len(satisfaction_df)
            satisfaction_rate = (
                ((satisfaction_df[metric_name] >= 1).sum() / total_entries) * 100
                if total_entries > 0 else 0
            )
            table_data.append([
                f'Priority {priority_idx} Satisfaction Rate (%)',
                f'{satisfaction_rate:.2f}',
            ])

            # Calculate violation period statistics
            violation_periods = calculate_violation_periods(
                satisfaction_df, metric_name, app_state.cfg.interaction.obs_period,
            )
            period_stats = get_violation_period_stats(violation_periods, cfg.violation_period_percentiles)

            # Add violation period statistics to table
            for stat_name, stat_value in period_stats.items():
                table_data.append([f'Priority {priority_idx} {stat_name}', stat_value])

    # Check for optimization performance metric (last priority should be optimization)
    if isinstance(priorities[-1], Optimization):
        opt_performance_df = app_state.metrics.read(
            metric='optimization_performance',
            start_time=start_time,
            end_time=end_time,
        )

        if not opt_performance_df.empty:
            table_data.append([
                'Average Optimization Performance',
                str(opt_performance_df['optimization_performance'].mean()),
            ])

    # Generate table and save
    table_str = tabulate(table_data, headers=headers, tablefmt='grid')
    (output_path / 'goal_violations.txt').write_text(table_str, encoding='utf-8')


def make_transition_statistics_table(
        cfg: ReportConfig,
        transitions: list[Transition],
        output_path: Path,
        app_state: AppState,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
    """
    Generate transition statistics table and save to file.
    """

    log.info("Generating transition statistics...")

    # Calculate contiguous sequence lengths
    sequence_lengths = calculate_contiguous_sequence_lengths(
        transitions,
        cfg.contiguous_time_threshold,
    )

    # Prepare table data
    table_data: list[list[str]] = [
        ['start time of report gen.', str(start_time)],
        ['end time of report gen.', str(end_time)],
    ]

    headers = [
        'Metric',
        'Value',
    ]

    # Basic statistics
    total_transitions = len(transitions)
    table_data.append(['Total Transitions', str(total_transitions)])

    # Add transitions filtered count from metrics if available
    try:
        transitions_filtered_df = app_state.metrics.read(
            metric='transitions_filtered',
            start_time=start_time,
            end_time=end_time,
        )
        if not transitions_filtered_df.empty:
            total_filtered = transitions_filtered_df['transitions_filtered'].sum()
            table_data.append(['Total Transitions Filtered', str(total_filtered)])

    except Exception as e:
        log.warning(f"Could not read transitions_filtered metric: {e}")

    # Sequence statistics
    sequence_stat_dict = get_sequence_stats(cfg, sequence_lengths)
    for k, v in sequence_stat_dict.items():
        table_data.append([k, str(v)])

    # Generate table and save
    table_str = tabulate(table_data, headers=headers, tablefmt='grid')
    (output_path / 'transition_statistics.txt').write_text(table_str, encoding='utf-8')


def generate_report(
        cfg: ReportConfig,
        data: list[pd.DataFrame],
        stages: list[StageCode],
        app_state: AppState,
        start_time: datetime,
        end_time: datetime,
        transitions: list[Transition] | None = None,
    ) -> None:

    output_path = Path(cfg.output_dir)
    if output_path.exists():
        log.warning(f'Output path {output_path} already exists. Deleting...')
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    make_stat_table(cfg, data, stages, output_path)
    make_distribution_plots(cfg, data, stages, output_path / 'plots')
    make_cross_correlation_table(cfg, data, stages, output_path)

    # Generate goal violations table
    make_goal_violations_table(
        cfg,
        output_path,
        app_state,
        start_time,
        end_time,
    )

    # Generate transition statistics if transitions are provided
    if transitions:
        make_transition_statistics_table(
            cfg,
            transitions,
            output_path,
            app_state,
            start_time,
            end_time,
        )
