import datetime as dt
import logging
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import DataMode
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.pipeline import ColumnDescriptions, Pipeline, PipelineReturn
from corerl.environment.async_env.async_env import DepAsyncEnvConfig
from corerl.eval.actor_critic import ActorCriticEval
from corerl.eval.monte_carlo import MonteCarloEvaluator
from corerl.state import AppState
from corerl.utils.time import split_into_chunks

log = logging.getLogger(__name__)

def load_entire_dataset(
    cfg: MainConfig,
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None
) -> pd.DataFrame:

    assert isinstance(cfg.env, DepAsyncEnvConfig)
    data_reader = DataReader(db_cfg=cfg.env.db)
    if start_time is None or end_time is None:
        time_stats = data_reader.get_time_stats()
        if start_time is None:
            start_time = time_stats.start
        if end_time is None:
            end_time = time_stats.end

    tag_names = [tag_cfg.name for tag_cfg in cfg.pipeline.tags]
    obs_period = cfg.interaction.obs_period
    data = data_reader.batch_aggregated_read(
        names=tag_names,
        start_time=start_time,
        end_time=end_time,
        bucket_width=obs_period,
        aggregation=cfg.env.db.data_agg,
    )
    return data


class OfflineTraining:
    def __init__(
        self,
        cfg: MainConfig
    ):
        self.cfg = cfg
        self.start_time = cfg.experiment.start_time
        self.end_time = cfg.experiment.end_time
        self.offline_steps = self.cfg.experiment.offline_steps
        self.pipeline_out: PipelineReturn | None = None

    def load_offline_transitions(self, pipeline: Pipeline):
        # Configure DataReader
        assert isinstance(self.cfg.env, DepAsyncEnvConfig)
        data_reader = DataReader(db_cfg=self.cfg.env.db)

        # Infer missing start or end time
        self.start_time, self.end_time = get_data_start_end_times(data_reader, self.start_time, self.end_time)

        # chunk offline reads
        chunk_width = self.cfg.experiment.pipeline_batch_duration
        time_chunks = split_into_chunks(self.start_time, self.end_time, width=chunk_width)

        # Pass offline data through data pipeline chunk by chunk to produce transitions
        tag_names = [tag_cfg.name for tag_cfg in self.cfg.pipeline.tags]
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

        ac_eval = ActorCriticEval(self.cfg.eval_cfgs.actor_critic, app_state, pipeline, agent, column_desc)
        ac_eval.get_test_states(self.pipeline_out.transitions)
        mc_eval = MonteCarloEvaluator(self.cfg.eval_cfgs.monte_carlo, app_state, agent)

        agent.load_buffer(self.pipeline_out)
        for buffer_name, size in agent.get_buffer_sizes().items():
            log.info(f"Agent {buffer_name} replay buffer size(s)", size)

        q_losses: list[float] = []
        pbar = tqdm(range(self.offline_steps))
        for i in pbar:
            if i in self.cfg.experiment.offline_eval_iters:
                mc_eval.execute_offline(i, self.pipeline_out)
                ac_eval.execute_offline(i)

            critic_loss = agent.update()
            q_losses += critic_loss

        return q_losses


def get_data_start_end_times(
    data_reader: DataReader, start_time: dt.datetime | None = None, end_time: dt.datetime | None = None
) -> Tuple[dt.datetime, dt.datetime]:
    if start_time is None or end_time is None:
        time_stats = data_reader.get_time_stats()
        if start_time is None:
            start_time = time_stats.start
        if end_time is None:
            end_time = time_stats.end

    return start_time, end_time
