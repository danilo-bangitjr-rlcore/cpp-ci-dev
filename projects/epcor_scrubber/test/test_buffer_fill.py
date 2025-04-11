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
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.messages.event_bus import EventBus
from corerl.state import AppState
from corerl.utils.maybe import Maybe
from corerl.utils.time import split_into_chunks

logger = logging.getLogger(__name__)


def test_buffer_load():
    cfg = direct_load_config(
        MainConfig, base="projects/epcor_scrubber/configs/", config_name="epcor_scrubber_local.yaml"
    )
    assert isinstance(cfg, MainConfig)

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    app_state = AppState(
        cfg=cfg,
        metrics=MetricsTable(cfg.metrics),
        evals=EvalsTable(cfg.evals),
        event_bus=EventBus(cfg.event_bus, cfg.env),
    )

    pipeline = Pipeline(app_state, cfg.pipeline)
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
