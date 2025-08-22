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

def load_entire_dataset(
    cfg: MainConfig,
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None,
) -> pd.DataFrame:

    assert isinstance(cfg.env, AsyncEnvConfig)
    data_reader = DataReader(db_cfg=cfg.env.db)
    if start_time is None or end_time is None:
        time_stats = data_reader.get_time_stats()
        if start_time is None:
            start_time = time_stats.start
        if end_time is None:
            end_time = time_stats.end

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
        data_reader: DataReader,
    ):

    # Infer missing start or end time
    offline_cfg = app_state.cfg.offline
    start_time, end_time = get_data_start_end_times(
        data_reader,
        offline_cfg.offline_start_time,
        offline_cfg.offline_end_time,
    )

    time_chunks = split_into_chunks(start_time, end_time, width=offline_cfg.pipeline_batch_duration)

    # Filter out evaluation periods if configured
    if offline_cfg.remove_eval_from_train and offline_cfg.eval_periods:
        time_chunks = exclude_from_chunks(time_chunks, offline_cfg.eval_periods)

    # Pass offline data through data pipeline chunk by chunk to produce transitions
    out = None
    tag_names = [tag_cfg.name for tag_cfg in get_scada_tags(app_state.cfg.pipeline.tags)]
    for chunk_start, chunk_end in time_chunks:
        chunk_data = data_reader.batch_aggregated_read(
            names=tag_names,
            start_time=chunk_start,
            end_time=chunk_end,
            bucket_width=app_state.cfg.interaction.obs_period,
            aggregation=app_state.cfg.env.db.data_agg,
        )
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
        cfg: MainConfig,
        app_state: AppState,
        agent: GreedyAC,
        pipeline: Pipeline,
        data_reader: DataReader,
    ):
    """
    Gives the data specfied in offline_cfg.eval_periods to the agent to get the agent's recommendations
    """
    if cfg.offline.eval_periods is None:
        log.info("No evaluation phase.")
        return

    tag_names = [tag_cfg.name for tag_cfg in get_scada_tags(cfg.pipeline.tags)]

    for eval_period in cfg.offline.eval_periods:
        start = eval_period[0]
        end = eval_period[1]
        log.info(f"Starting evaluation phase: {start} to {end}")
        params = OfflineRecParameters(
            start,
            end,
            cfg.interaction.obs_period,
            tag_names=tag_names,
            data_agg=cfg.env.db.data_agg,
            update_agent=cfg.offline.update_agent_during_offline_recs,
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
    # Create 1 obs_period-wide chunks
    time_chunks = split_into_chunks(
        params.eval_start,
        params.eval_end,
        width=params.obs_period,
    )
    for chunk_start, chunk_end in time_chunks:
        log.info(f"Rolling out on {chunk_start} to {chunk_end}.")
        chunk_data = data_reader.batch_aggregated_read(
            names=params.tag_names,
            start_time=chunk_start,
            end_time=chunk_end,
            bucket_width=params.obs_period,
            aggregation=params.data_agg,
        )
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
                norm_next_a_df = norm_next_a_df .clip(lower=np.asarray(state.a_lo), upper=np.asarray(state.a_hi))
                _write_to_metrics(app_state, norm_next_a_df, prefix="ACTION-RECOMMEND")

            _write_to_metrics(app_state, chunk_pr.states, prefix="STATE-")
            _write_to_metrics(app_state, chunk_pr.rewards)
            _write_to_metrics(app_state, chunk_pr.actions, prefix="ACTION-")
            state = get_latest_state(chunk_pr)
            agent_eval.q_values_and_act_prob(app_state, agent, state.features)

        if chunk_pr.transitions is None or len(chunk_pr.transitions) == 0:
            log.warning(f"No transitions found for eval chunk: {chunk_start} to {chunk_end}")
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


def get_data_start_end_times(
    data_reader: DataReader, start_time: dt.datetime | None = None, end_time: dt.datetime | None = None,
) -> tuple[dt.datetime, dt.datetime]:
    if start_time is None or end_time is None:
        time_stats = data_reader.get_time_stats()
        if start_time is None:
            start_time = time_stats.start
            if end_time is not None:
                assert (
                    start_time < end_time
                ), "The specified 'end' timestamp must come after the first timestamp in the offline data."
        if end_time is None:
            end_time = time_stats.end
            if start_time != time_stats.start:
                assert (
                    start_time < end_time
                ), "The specified 'start' timestamp must come before the final timestamp in the offline data."

    return start_time, end_time

