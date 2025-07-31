import datetime as dt
import logging

import jax
import jax.numpy as jnp
import pandas as pd

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import DataMode
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.pipeline import ColumnDescriptions, Pipeline, PipelineReturn
from corerl.environment.async_env.async_env import AsyncEnvConfig
from corerl.eval.agent import offline_q_values_and_act_prob
from corerl.eval.plotting.report import make_actor_critic_plots
from corerl.state import AppState
from corerl.tags.tag_config import get_scada_tags
from corerl.utils.time import split_into_chunks

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


class OfflineTraining:
    def __init__(
        self,
        cfg: MainConfig,
    ):
        self.cfg = cfg
        self.offline_eval_iters = cfg.offline.offline_eval_iters
        self.start_time = cfg.offline.offline_start_time
        self.end_time = cfg.offline.offline_end_time
        self.offline_steps = self.cfg.offline.offline_steps
        self.pipeline_out: PipelineReturn | None = None

    def load_offline_transitions(self, pipeline: Pipeline):
        # Configure DataReader
        assert isinstance(self.cfg.env, AsyncEnvConfig)
        data_reader = DataReader(db_cfg=self.cfg.env.db)

        # Infer missing start or end time
        self.start_time, self.end_time = get_data_start_end_times(data_reader, self.start_time, self.end_time)

        # chunk offline reads
        chunk_width = self.cfg.offline.pipeline_batch_duration
        time_chunks = split_into_chunks(self.start_time, self.end_time, width=chunk_width)

        # Pass offline data through data pipeline chunk by chunk to produce transitions
        tag_names = [tag_cfg.name for tag_cfg in get_scada_tags(self.cfg.pipeline.tags)]
        for chunk_start, chunk_end in time_chunks:
            chunk_data = data_reader.batch_aggregated_read(
                names=tag_names,
                start_time=chunk_start,
                end_time=chunk_end,
                bucket_width=self.cfg.interaction.obs_period,
                aggregation=self.cfg.env.db.data_agg,
            )
            chunk_pr = pipeline(
                data=chunk_data,
                data_mode=DataMode.OFFLINE,
                reset_temporal_state=False,
            )

            if self.pipeline_out:
                self.pipeline_out += chunk_pr
            else:
                self.pipeline_out = chunk_pr

    def train(self, app_state: AppState, agent: GreedyAC, pipeline: Pipeline, column_desc: ColumnDescriptions):
        assert isinstance(self.start_time, dt.datetime)
        assert isinstance(self.end_time, dt.datetime)
        assert self.pipeline_out is not None
        assert self.pipeline_out.transitions is not None
        assert len(self.pipeline_out.transitions) > 0, (
            "You must first load offline transitions before you can perform offline training"
        )
        log.info("Starting offline agent training...")

        eval_states, eval_dates = get_one_day_per_month(self.pipeline_out, self.start_time, self.end_time)

        agent.update_buffer(self.pipeline_out)
        for buffer_name, size in agent.get_buffer_sizes().items():
            log.info(f"Agent {buffer_name} replay buffer size(s)", size)

        q_losses: list[float] = []
        for i in range(self.offline_steps):
            if i in self.offline_eval_iters:
                for j, eval_state in enumerate(eval_states):
                    x_axis_actions, probs, qs = offline_q_values_and_act_prob(app_state, agent, eval_state)
                    curr_date = eval_dates[int(j / 24)]
                    curr_date_hour = curr_date.replace(hour=j%24)
                    make_actor_critic_plots(str(curr_date_hour), x_axis_actions, probs, qs, i, j, self.cfg.save_path)

            critic_loss = agent.update()
            q_losses += critic_loss

        return q_losses


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

def get_one_day_per_month(
    pr: PipelineReturn,
    start: dt.datetime,
    end: dt.datetime,
) -> tuple[list[jax.Array], list[dt.datetime]]:
    """
    Picks one day each month in the historical dataset and returns each state, datetime observed on the hour mark
    """
    eval_states = []
    eval_dates = []

    start_day = 28
    curr_time = start.replace(day=start_day, hour=0, minute=0, second=0)
    while curr_time < end:
        day_states = []
        hour = 0
        # Day in month determined once it's confirmed that there's a state for each hour within the given day
        while len(day_states) < 24:
            curr_time = curr_time.replace(hour=hour)
            curr_state = jnp.asarray(pr.states.loc[str(curr_time)].to_numpy())
            # If there's a NaN within a state, we move on to the next day
            if jnp.isnan(curr_state).any():
                day_states = []
                hour = 0
                if curr_time.day == 1:
                    break
                curr_time = curr_time.replace(day=curr_time.day-1)
                continue
            day_states.append(jnp.asarray(pr.states.loc[str(curr_time)].to_numpy()))
            hour += 1

        if len(day_states) > 0:
            eval_states += day_states
            eval_dates.append(curr_time)

        if curr_time.month == 12:
            curr_time = curr_time.replace(year=curr_time.year+1, month=1, day=start_day)
        else:
            curr_time = curr_time.replace(month=curr_time.month+1, day=start_day)

    return eval_states, eval_dates
