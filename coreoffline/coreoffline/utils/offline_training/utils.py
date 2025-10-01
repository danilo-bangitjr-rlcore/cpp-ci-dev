import logging
from datetime import datetime

import jax.numpy as jnp
import numpy as np
import pandas as pd
from corerl.agent.greedy_ac import GreedyAC
from corerl.data_pipeline.pipeline import Pipeline, PipelineReturn
from corerl.eval import agent as agent_eval
from corerl.state import AppState
from lib_agent.buffer.buffer import State
from lib_agent.buffer.datatypes import DataMode

from coreoffline.utils.config import OfflineMainConfig
from coreoffline.utils.data_loading import load_data_chunks

log = logging.getLogger(__name__)


def get_all_offline_recommendations(
    app_state: AppState,
    agent: GreedyAC,
    pipeline: Pipeline,
):
    """
    Gives the data specfied in offline_cfg.eval_periods to the agent to get the agent's recommendations
    """

    assert isinstance(app_state.cfg, OfflineMainConfig)
    if app_state.cfg.offline_training.eval_periods is None:
        log.info("No evaluation phase.")
        return

    for eval_period in app_state.cfg.offline_training.eval_periods:
        start = eval_period[0]
        end = eval_period[1]
        log.info(f"Starting evaluation phase: {start} to {end}")

        get_offline_recommendations(app_state, agent, pipeline, start, end)


def get_offline_recommendations(
    app_state: AppState,
    agent: GreedyAC,
    pipeline: Pipeline,
    start: datetime,
    end: datetime,
):

    assert isinstance(app_state.cfg, OfflineMainConfig)
    state = None

    data_chunks = load_data_chunks(
        cfg=app_state.cfg,
        start_time=start,
        end_time=end,
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

        if app_state.cfg.offline_training.update_agent_during_offline_recs:
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
