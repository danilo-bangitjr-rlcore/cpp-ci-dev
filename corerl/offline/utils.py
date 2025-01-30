import datetime as dt
import logging
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from corerl.agent.base import BaseAgent
from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import CallerCode
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.pipeline import ColumnDescriptions, Pipeline, PipelineReturn
from corerl.eval.actor_critic import ActorCriticEval
from corerl.eval.monte_carlo import MonteCarloEvaluator
from corerl.state import AppState
from corerl.utils.time import split_into_chunks

log = logging.getLogger(__name__)

def load_entire_dataset(
        cfg: MainConfig,
        start_time: dt.datetime | None = None, end_time: dt.datetime | None = None
        ) -> pd.DataFrame:

    data_reader = DataReader(db_cfg=cfg.pipeline.db)
    if start_time is None or end_time is None:
        time_stats = data_reader.get_time_stats()
        if start_time is None:
            start_time = time_stats.start
        if end_time is None:
            end_time = time_stats.end

    tag_names = [tag_cfg.name for tag_cfg in cfg.pipeline.tags]
    obs_period = cfg.pipeline.obs_period
    data = data_reader.batch_aggregated_read(
        names=tag_names,
        start_time=start_time,
        end_time=end_time,
        bucket_width=obs_period,
        aggregation=cfg.pipeline.db.data_agg,
    )
    return data


class OfflineTraining:
    def __init__(self, cfg: MainConfig):
        self.cfg = cfg
        self.offline_steps = self.cfg.experiment.offline_steps
        self.pipeline_out: PipelineReturn | None = None

    def load_offline_transitions(
        self, pipeline: Pipeline, start_time: dt.datetime | None = None, end_time: dt.datetime | None = None
    ):
        # Configure DataReader
        data_reader = DataReader(db_cfg=self.cfg.pipeline.db)

        # Infer missing start or end time
        start_time, end_time = get_data_start_end_times(data_reader, start_time, end_time)

        # chunk offline reads
        chunk_width = self.cfg.experiment.pipeline_batch_duration_days
        time_chunks = split_into_chunks(start_time, end_time, width=dt.timedelta(chunk_width))

        # Pass offline data through data pipeline chunk by chunk to produce transitions
        tag_names = [tag_cfg.name for tag_cfg in self.cfg.pipeline.tags]
        for chunk_start, chunk_end in time_chunks:
            chunk_data = data_reader.batch_aggregated_read(
                names=tag_names,
                start_time=chunk_start,
                end_time=chunk_end,
                bucket_width=self.cfg.pipeline.obs_period,
                aggregation=self.cfg.pipeline.db.data_agg,
            )
            chunk_pr = pipeline(
                data=chunk_data,
                caller_code=CallerCode.OFFLINE,
                reset_temporal_state=False,
            )

            if self.pipeline_out:
                self.pipeline_out += chunk_pr
            else:
                self.pipeline_out = chunk_pr

    def train(self, app_state: AppState, agent: BaseAgent, column_desc: ColumnDescriptions):
        assert self.pipeline_out is not None
        assert self.pipeline_out.transitions is not None
        assert len(self.pipeline_out.transitions) > 0, (
            "You must first load offline transitions before you can perform offline training"
        )
        log.info("Starting offline agent training...")

        ac_eval = ActorCriticEval(self.cfg.eval_cfgs.actor_critic, app_state, agent, column_desc)
        ac_eval.get_test_states(self.pipeline_out.transitions)
        mc_eval = MonteCarloEvaluator(self.cfg.eval_cfgs.monte_carlo, app_state, agent, self.pipeline_out)

        agent.load_buffer(self.pipeline_out)
        for buffer_name, size in agent.get_buffer_sizes().items():
            log.info(f"Agent {buffer_name} replay buffer size(s)", size)

        q_losses: list[float] = []
        pbar = tqdm(range(self.offline_steps))
        for i in pbar:
            mc_eval(i)
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
