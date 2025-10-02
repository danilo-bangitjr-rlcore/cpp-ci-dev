import datetime as dt
import logging
from datetime import timedelta
from math import ceil

import numpy as np
from corerl.agent.greedy_ac import GreedyAC
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.pipeline import Pipeline, PipelineReturn
from corerl.environment.async_env.async_env import AsyncEnvConfig
from corerl.state import AppState
from corerl.tags.tag_config import get_scada_tags
from corerl.utils.time import exclude_from_chunks, split_into_chunks
from lib_agent.buffer.datatypes import DataMode
from lib_progress.tracker import ProgressTracker, track

from coreoffline.utils.config import OfflineMainConfig

log = logging.getLogger(__name__)


def get_data_reader(cfg: OfflineMainConfig):
    """Get DataReader instance from config"""
    assert isinstance(cfg.env, AsyncEnvConfig)
    return DataReader(db_cfg=cfg.env.db)


def get_time_range(
    data_reader: DataReader,
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None,
):
    """
    Get start and end times, using database time stats for None values.
    """
    if start_time is not None and end_time is not None:
        return start_time, end_time

    time_stats = data_reader.get_time_stats()

    if start_time is None:
        start_time = time_stats.start
    if end_time is None:
        end_time = time_stats.end

    # Validation
    assert start_time < end_time, (
        "Start time must come before end time. "
        f"Got start: {start_time}, end: {end_time}"
    )

    return start_time, end_time


def load_data_chunks(
    cfg: OfflineMainConfig,
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None,
    chunk_duration: timedelta | None = None,
    exclude_periods: list[tuple[dt.datetime, dt.datetime]] | None = None,
):
    """
    Returns a generator that yields chunks of data and the total number of chunks.
    """
    data_reader = get_data_reader(cfg)
    start_time, end_time = get_time_range(data_reader, start_time, end_time)
    # Use config default if no chunk_duration provided
    if chunk_duration is None:
        chunk_duration = cfg.offline_training.pipeline_batch_duration

    tag_names = [tag_cfg.name for tag_cfg in get_scada_tags(cfg.pipeline.tags)]
    obs_period = cfg.interaction.obs_period

    time_chunks = split_into_chunks(start_time, end_time, width=chunk_duration)

    # Filter out excluded periods if configured
    if exclude_periods:
        time_chunks = exclude_from_chunks(time_chunks, exclude_periods)
    time_chunks = list(time_chunks)
    num_chunks = len(time_chunks)

    def chunk_generator():
        try:
            for chunk_start, chunk_end in time_chunks:
                chunk_data = data_reader.batch_aggregated_read(
                    names=tag_names,
                    start_time=chunk_start,
                    end_time=chunk_end,
                    bucket_width=obs_period,
                    aggregation=cfg.env.db.data_agg,
                )
                if not chunk_data.empty:
                    yield chunk_data
        finally:
            data_reader.close()

    return chunk_generator(), num_chunks


def load_entire_dataset(
    cfg: OfflineMainConfig,
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None,
):

    data_reader = get_data_reader(cfg)
    start_time, end_time = get_time_range(data_reader, start_time, end_time)

    tag_names = [tag_cfg.name for tag_cfg in get_scada_tags(cfg.pipeline.tags)]
    obs_period = cfg.interaction.obs_period
    out = data_reader.batch_aggregated_read(
        names=tag_names,
        start_time=start_time,
        end_time=end_time,
        bucket_width=obs_period,
        aggregation=cfg.env.db.data_agg,
    )
    data_reader.close()
    return out


def offline_rl_from_buffer(agent: GreedyAC, steps: int = 100):
    log.info("Starting offline agent training...")

    for buffer_name, size_list in agent.get_buffer_sizes().items():
        log.info(f"Agent {buffer_name} replay buffer size(s): {size_list}")

    q_losses: list[float] = []

    with ProgressTracker(total=steps, desc='Offline agent training', update_interval=10) as tracker:
        for _ in range(steps):
            critic_loss = agent.update()
            q_losses += critic_loss
            tracker.update(
                metrics={"q_loss": float(q_losses[-1])},
            )

    return q_losses


def load_offline_transitions(app_state: AppState, pipeline: Pipeline):
    """
    Load offline transitions from database through the data pipeline.
    """
    assert isinstance(app_state.cfg, OfflineMainConfig)
    # Get time range from config
    offline_cfg = app_state.cfg.offline_training

    # Get configuration for data loading
    exclude_periods = offline_cfg.eval_periods if offline_cfg.remove_eval_from_train else None

    # Pass offline data through data pipeline chunk by chunk to produce transitions
    out = None

    data_chunks, num_chunks = load_data_chunks(
        cfg=app_state.cfg,
        start_time=offline_cfg.offline_start_time,
        end_time=offline_cfg.offline_end_time,
        exclude_periods=exclude_periods,
    )

    for chunk_data in track(
        data_chunks,
        desc="Loading data by chunk",
        total=num_chunks,
        update_interval=5,
    ):
        chunk_pr = pipeline(
            data=chunk_data,
            data_mode=DataMode.OFFLINE,
            reset_temporal_state=False,
        )

        if out:
            out += chunk_pr
        else:
            out = chunk_pr

    # Apply test split if requested
    test_split = offline_cfg.test_split
    if test_split > 0.0 and out is not None and out.transitions is not None:
        return get_test_split(out, test_split)

    return out, []


def get_test_split(pr: PipelineReturn, test_split: float):
    transitions = pr.transitions
    assert transitions is not None
    n_transitions = len(transitions)
    n_test = ceil(n_transitions * test_split)

    # Randomly select test indices
    test_indices = np.random.choice(n_transitions, size=n_test, replace=False)
    test_mask = np.zeros(n_transitions, dtype=bool)
    test_mask[test_indices] = True

    # Split transitions
    test_transitions = [transitions[i] for i in range(n_transitions) if test_mask[i]]
    train_transitions = [transitions[i] for i in range(n_transitions) if not test_mask[i]]

    pr.transitions = train_transitions

    return pr, test_transitions
