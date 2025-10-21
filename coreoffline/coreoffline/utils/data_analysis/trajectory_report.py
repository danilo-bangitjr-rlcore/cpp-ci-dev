import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from corerl.state import AppState
from lib_agent.buffer.datatypes import Trajectory

from coreoffline.utils.config import ReportConfig
from coreoffline.utils.data_analysis.data_report import log, save_as_csv, save_as_txt


def are_trajectories_contiguous(
    prev_trajectory: Trajectory,
    curr_trajectory: Trajectory,
    time_threshold: timedelta,
):
    """
    Determine if two trajectories are contiguous based on timestamp proximity.
    """
    # Check if both trajectories have timestamps
    prev_timestamp = prev_trajectory.end_time
    curr_timestamp = curr_trajectory.start_time

    if prev_timestamp is None or curr_timestamp is None:
        return False

    # Calculate time difference
    time_diff = curr_timestamp - prev_timestamp
    return time_diff <= time_threshold


def calculate_contiguous_sequence_lengths(
    trajectories: list[Trajectory],
    time_threshold: timedelta,
):
    """
    Calculate lengths of contiguous trajectory sequences.
    Trajectories are considered contiguous if the time gap between consecutive trajectories
    is less than the specified threshold.
    """
    if not trajectories:
        return []

    sequence_lengths = []
    current_length = 1

    for i in range(1, len(trajectories)):
        prev_trajectory = trajectories[i - 1]
        curr_trajectory = trajectories[i]

        # Check if trajectories are contiguous based on timestamp
        if are_trajectories_contiguous(prev_trajectory, curr_trajectory, time_threshold):
            current_length += 1
        else:
            sequence_lengths.append(current_length)
            current_length = 1

    # Add the last sequence
    sequence_lengths.append(current_length)

    return sequence_lengths


def get_sequence_stats(cfg: ReportConfig, sequence_lengths: list[int]):
    return_dict = {
        'Total Sequences': len(sequence_lengths),
        'Average Sequence Length': np.mean(sequence_lengths),
        'Min Sequence Length': np.min(sequence_lengths),
        'Max Sequence Length': np.max(sequence_lengths),
    }
    # Add percentiles
    for p in cfg.trajectory_percentiles:
        percentile_value = np.percentile(sequence_lengths, p * 100)
        return_dict[f'P{int(p * 100)} Sequence Length'] = percentile_value
    return return_dict


def make_trajectory_statistics_table(
    cfg: ReportConfig,
    trajectories: list[Trajectory],
    output_path: Path,
    app_state: AppState,
    start_time: datetime,
    end_time: datetime,
):
    """
    Generate trajectory statistics table and save to file.
    """

    log.info("Generating trajectory statistics...")

    # Calculate contiguous sequence lengths
    sequence_lengths = calculate_contiguous_sequence_lengths(
        trajectories,
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
    total_trajectories = len(trajectories)
    table_data.append(['Total Trajectories', str(total_trajectories)])

    # Add trajectories filtered count from metrics if available
    try:
        trajectories_filtered_df = app_state.metrics.read(
            metric='trajectories_filtered',
            start_time=start_time,
            end_time=end_time,
            prefix_match=True,
        )

        for col in trajectories_filtered_df.columns:
            if col.startswith("trajectories_filtered"):
                total_filtered = trajectories_filtered_df[col].sum()
                trajectory_filter = col.removeprefix("trajectories_filtered_by_")
                table_data.append([f'Total Trajectories Filtered by {trajectory_filter}', str(total_filtered)])

    except Exception as e:
        log.warning(f"Could not read trajectories_filtered metric: {e}")

    # Sequence statistics
    sequence_stat_dict = get_sequence_stats(cfg, sequence_lengths)
    for k, v in sequence_stat_dict.items():
        table_data.append([k, str(v)])

    # Generate table and save
    full_table_data = [headers, *table_data]
    save_as_txt(full_table_data, output_path, 'trajectory_statistics')
    save_as_csv(full_table_data, output_path, 'trajectory_statistics')


def generate_report(
    cfg: ReportConfig,
    app_state: AppState,
    start_time: datetime,
    end_time: datetime,
    trajectories: list[Trajectory],
):

    output_path = Path(cfg.output_dir)
    if output_path.exists():
        log.warning(f'Output path {output_path} already exists. Deleting...')
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    make_trajectory_statistics_table(
        cfg,
        trajectories,
        output_path,
        app_state,
        start_time,
        end_time,
    )
