import logging
import random
from datetime import UTC

import numpy as np
import torch

from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config
from corerl.data_pipeline.datatypes import DataMode
from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.async_env.async_env import DepAsyncEnvConfig
from corerl.environment.registry import register_custom_envs
from corerl.eval.evals import evals_group
from corerl.eval.metrics import metrics_group
from corerl.interaction.configs import DepInteractionConfig
from corerl.messages.event_bus import EventBus
from corerl.state import AppState
from corerl.utils.maybe import Maybe
from corerl.utils.time import split_into_chunks

logger = logging.getLogger(__name__)


def test_buffer_load():
    cfg = direct_load_config(
        MainConfig, base="projects/epcor_scrubber/configs/", config_name="epcor_scrubber_local.yaml"
    )

    # get custom gym environments
    register_custom_envs()

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    app_state = AppState(
        cfg=cfg,
        metrics=metrics_group.dispatch(cfg.metrics),
        evals=evals_group.dispatch(cfg.evals),
        event_bus=EventBus(cfg.event_bus, cfg.env),
    )

    pipeline = Pipeline(app_state, cfg.pipeline)
    assert isinstance(cfg.env, DepAsyncEnvConfig)
    assert isinstance(cfg.interaction, DepInteractionConfig)
    data_reader = DataReader(db_cfg=cfg.env.db)
    time_stats = data_reader.get_time_stats()
    chunk_start = Maybe(cfg.interaction.hist_chunk_start).or_else(time_stats.start).astimezone(UTC)
    chunk_end = time_stats.end
    chunks = split_into_chunks(
        chunk_start,
        chunk_end,
        width=cfg.interaction.obs_period * cfg.interaction.historical_batch_size
    )

    start_time, end_time = next(chunks)
    tag_names = [tag_cfg.name for tag_cfg in pipeline.tags]
    tag_aggs = {tag.name: tag.agg for tag in pipeline.tags}
    chunk_data = data_reader.batch_aggregated_read(
        names=tag_names,
        start_time=start_time,
        end_time=end_time,
        bucket_width=cfg.interaction.obs_period,
        tag_aggregations=tag_aggs
    )
    logger.info(f"Loading chunk data from {chunk_data.index[0]} to {chunk_data.index[-1]}")

    pipeline_out = pipeline(
        data=chunk_data,
        data_mode=DataMode.OFFLINE,
        reset_temporal_state=False,
    )

    assert pipeline_out.transitions is not None
    assert len(pipeline_out.transitions) > 1000
