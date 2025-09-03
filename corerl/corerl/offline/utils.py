import datetime as dt
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import jax.numpy as jnp
import numpy as np
import pandas as pd
from lib_agent.buffer.buffer import State

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import DataMode
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.pipeline import Pipeline, PipelineReturn
from corerl.environment.async_env.async_env import AsyncEnvConfig
from corerl.eval import agent as agent_eval
from corerl.state import AppState
from corerl.tags.components.opc import Agg
from corerl.tags.tag_config import get_scada_tags
from corerl.utils.time import exclude_from_chunks, split_into_chunks

log = logging.getLogger(__name__)


def get_data_reader(cfg: MainConfig) -> DataReader:
    """Get DataReader instance from config"""
    assert isinstance(cfg.env, AsyncEnvConfig)
    return DataReader(db_cfg=cfg.env.db)


def get_time_range(
    data_reader: DataReader,
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None,
) -> tuple[dt.datetime, dt.datetime]:
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
    cfg: MainConfig,
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None,
    chunk_duration: timedelta | None = None,
    exclude_periods: list[tuple[dt.datetime, dt.datetime]] | None = None,
):
    """
    Generator that yields chunks of data from the database.
    """
    data_reader = get_data_reader(cfg)
    start_time, end_time = get_time_range(data_reader, start_time, end_time)
    # Use config default if no chunk_duration provided
    if chunk_duration is None:
        chunk_duration = cfg.offline.pipeline_batch_duration

    tag_names = [tag_cfg.name for tag_cfg in get_scada_tags(cfg.pipeline.tags)]
    obs_period = cfg.interaction.obs_period

    time_chunks = split_into_chunks(start_time, end_time, width=chunk_duration)

    # Filter out excluded periods if configured
    if exclude_periods:
        time_chunks = exclude_from_chunks(time_chunks, exclude_periods)

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


def load_entire_dataset(
    cfg: MainConfig,
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None,
) -> pd.DataFrame:

    data_reader = get_data_reader(cfg)
    start_time, end_time = get_time_range(data_reader, start_time, end_time)

    tag_names = [tag_cfg.name for tag_cfg in get_scada_tags(cfg.pipeline.tags)]
    obs_period = cfg.interaction.obs_period
    return data_reader.batch_aggregated_read(
        names=tag_names,
        start_time=start_time,
        end_time=end_time,
        bucket_width=obs_period,
        aggregation=cfg.env.db.data_agg,
    )


def offline_rl_from_buffer(agent: GreedyAC, steps: int=100):
    log.info("Starting offline agent training...")

    for buffer_name, size_list in agent.get_buffer_sizes().items():
        log.info(f"Agent {buffer_name} replay buffer size(s): {size_list}")

    q_losses: list[float] = []
    for step in range(steps):
        critic_loss = agent.update()
        q_losses += critic_loss
        if step % 10 == 0 or step == steps - 1:
            log.info(f"Offline agent training step {step}/{steps}, last loss: {q_losses[-1]}")

    return q_losses

def load_offline_transitions(
        app_state: AppState,
        pipeline: Pipeline,
    ):
    """
    Load offline transitions from database through the data pipeline.
    """
    # Get time range from config
    offline_cfg = app_state.cfg.offline

    # Get configuration for data loading
    exclude_periods = offline_cfg.eval_periods if offline_cfg.remove_eval_from_train else None

    # Pass offline data through data pipeline chunk by chunk to produce transitions
    out = None

    data_chunks = load_data_chunks(
        cfg=app_state.cfg,
        start_time=offline_cfg.offline_start_time,
        end_time=offline_cfg.offline_end_time,
        exclude_periods=exclude_periods,
    )

    for chunk_data in data_chunks:
        chunk_pr = pipeline(
            data=chunk_data,
            data_mode=DataMode.OFFLINE,
            reset_temporal_state=False,
        )

        if out:
            out += chunk_pr
        else:
            out = chunk_pr

    return out


def get_all_offline_recommendations(
        app_state: AppState,
        agent: GreedyAC,
        pipeline: Pipeline,
        data_reader: DataReader,
    ):
    """
    Gives the data specfied in offline_cfg.eval_periods to the agent to get the agent's recommendations
    """
    if app_state.cfg.offline.eval_periods is None:
        log.info("No evaluation phase.")
        return

    tag_names = [tag_cfg.name for tag_cfg in get_scada_tags(app_state.cfg.pipeline.tags)]

    for eval_period in app_state.cfg.offline.eval_periods:
        start = eval_period[0]
        end = eval_period[1]
        log.info(f"Starting evaluation phase: {start} to {end}")
        params = OfflineRecParameters(
            start,
            end,
            app_state.cfg.interaction.obs_period,
            tag_names=tag_names,
            data_agg=app_state.cfg.env.db.data_agg,
            update_agent=app_state.cfg.offline.update_agent_during_offline_recs,
        )
        get_offline_recommendations(app_state, agent, pipeline, data_reader, params)

@dataclass
class OfflineRecParameters:
    eval_start: datetime
    eval_end: datetime
    obs_period: timedelta
    tag_names: list[str]
    data_agg: Agg
    update_agent: bool = True

def get_offline_recommendations(
    app_state: AppState,
    agent: GreedyAC,
    pipeline: Pipeline,
    data_reader: DataReader,
    params: OfflineRecParameters,
    ):

    state = None

    data_chunks = load_data_chunks(
        cfg=app_state.cfg,
        start_time=params.eval_start,
        end_time=params._eval_end,
        chunk_duration=app_state.cfg.interaction.obs_period,  # 1 obs_period-wide chunks
        exclude_periods=None,
    )

    for chunk_data in data_chunks:
        log.info(f"Rolling out on chunk with {len(chunk_data)} rows.")
        chunk_pr = pipeline(
            data=chunk_data,
            data_mode=DataMode.ONLINE,
            reset_temporal_state=False,
        )

        if len(chunk_pr.states) > 0 and len(chunk_pr.action_lo) > 0 and len(chunk_pr.action_hi) > 0:
            if state is not None:
                recommended_action = agent.get_action_interaction(state)
                norm_next_a_df = pipeline.action_constructor.get_action_df(recommended_action)
                # clip to the normalized action bounds
                norm_next_a_df = norm_next_a_df.clip(lower=np.asarray(state.a_lo), upper=np.asarray(state.a_hi))
                _write_to_metrics(app_state, norm_next_a_df, prefix="ACTION-RECOMMEND")

            _write_to_metrics(app_state, chunk_pr.states, prefix="STATE-")
            _write_to_metrics(app_state, chunk_pr.rewards)
            _write_to_metrics(app_state, chunk_pr.actions, prefix="ACTION-")
            state = get_latest_state(chunk_pr)
            agent_eval.q_values_and_act_prob(app_state, agent, state.features)

        if chunk_pr.transitions is None or len(chunk_pr.transitions) == 0:
            log.warning("No transitions found for eval chunk")
            continue

        if params.update_agent:
            agent.update_buffer(chunk_pr)
            losses = agent.update()
            log.info(f"Agent updated with num transitions={len(chunk_pr.transitions)}, final q loss={losses[-1]}")

        app_state.agent_step += 1

def _write_to_metrics(app_state: AppState, df: pd.DataFrame, prefix: str = ""):
    for feat_name in df.columns:
        val = df[feat_name].values[0]
        app_state.metrics.write(
            agent_step=app_state.agent_step,
            metric=prefix + feat_name,
            value=val,
        )

def get_latest_state(pipe_return: PipelineReturn):
    return State(
        features=jnp.asarray(pipe_return.states.iloc[-1]),
        a_lo=jnp.asarray(pipe_return.action_lo.iloc[-1]),
        a_hi=jnp.asarray(pipe_return.action_hi.iloc[-1]),
        dp=jnp.ones((1,)),
        last_a=jnp.asarray(pipe_return.actions.iloc[-1]),
    )



