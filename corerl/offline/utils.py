import logging
log = logging.getLogger(__name__)

import datetime as dt
from tqdm import tqdm

from corerl.config import MainConfig
from corerl.agent.base import BaseAgent
from corerl.data_pipeline.datatypes import Transition, CallerCode
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.pipeline import Pipeline
from corerl.utils.time import split_into_chunks

def load_offline_transitions(cfg: MainConfig, pipeline: Pipeline):
    # Configure DataReader
    data_reader = DataReader(db_cfg=cfg.pipeline.db)
    db_time_stats = data_reader.get_time_stats()
    tag_names = [tag_cfg.name for tag_cfg in cfg.pipeline.tags]

    time_chunks = split_into_chunks(
        db_time_stats.start,
        db_time_stats.end,
        width=dt.timedelta(cfg.experiment.pipeline_batch_duration_days),
    )

    offline_transitions: list[Transition] = []
    for chunk_start, chunk_end in time_chunks:
        chunk_data = data_reader.batch_aggregated_read(
            names=tag_names,
            start_time=chunk_start,
            end_time=chunk_end,
            bucket_width=dt.timedelta(seconds=cfg.obs_period),
            aggregation=cfg.pipeline.db.data_agg,
        )
        pipeline_out = pipeline(
            data=chunk_data,
            caller_code=CallerCode.OFFLINE,
            reset_temporal_state=False,
        )

        if pipeline_out.transitions is not None:
            offline_transitions += pipeline_out.transitions

    return offline_transitions

def offline_training(
    cfg: MainConfig,
    agent: BaseAgent,
    train_transitions: list[Transition],
):
    log.info('Starting offline agent training...')

    agent.load_buffer(train_transitions)
    for buffer_name, size in agent.get_buffer_sizes().items():
        log.info(f"Agent {buffer_name} replay buffer size(s)", size)

    offline_steps = cfg.experiment.offline_steps
    q_losses: list[float] = []
    pbar = tqdm(range(offline_steps))
    for _ in pbar:
        critic_loss = agent.update()
        q_losses += critic_loss

    return q_losses
