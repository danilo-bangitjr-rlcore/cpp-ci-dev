

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from corerl.data_pipeline.datatypes import Transition
from corerl.state import AppState

from coreoffline.config import ReportConfig
from coreoffline.data_analysis.data_report import log, save_as_csv, save_as_txt


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


def make_transition_statistics_table(
    cfg: ReportConfig,
    transitions: list[Transition],
    output_path: Path,
    app_state: AppState,
    start_time: datetime,
    end_time: datetime,
):
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
            prefix_match=True,
        )

        for col in transitions_filtered_df.columns:
            if col.startswith("transitions_filtered"):
                total_filtered = transitions_filtered_df[col].sum()
                transition_filter = col.removeprefix("transitions_filtered_by_")
                table_data.append([f'Total Transitions Filtered by {transition_filter}', str(total_filtered)])

    except Exception as e:
        log.warning(f"Could not read transitions_filtered metric: {e}")

    # Sequence statistics
    sequence_stat_dict = get_sequence_stats(cfg, sequence_lengths)
    for k, v in sequence_stat_dict.items():
        table_data.append([k, str(v)])

    # Generate table and save
    full_table_data = [headers, *table_data]
    save_as_txt(full_table_data, output_path, 'transition_statistics')
    save_as_csv(full_table_data, output_path, 'transition_statistics')
