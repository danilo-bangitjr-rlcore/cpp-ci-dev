import logging
import random

import numpy as np
import torch

from corerl.config import MainConfig
from corerl.configs.loader import load_config
from corerl.data_pipeline.datatypes import DataMode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.data_report import generate_report
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.messages.event_bus import DummyEventBus
from corerl.offline.utils import load_entire_dataset
from corerl.state import AppState
from corerl.utils.device import device

log = logging.getLogger(__name__)


@load_config(MainConfig, base='config/')
def main(cfg: MainConfig):
    """
    Assuming offline data has already been written to TimescaleDB
    """
    device.update_device(cfg.experiment.device)

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    app_state = AppState(
        cfg,
        evals=EvalsTable(cfg.evals),
        metrics=MetricsTable(cfg.metrics),
        event_bus=DummyEventBus(),
    )

    pipeline = Pipeline(app_state, cfg.pipeline)
    log.info("loading dataset...")
    data = load_entire_dataset(cfg)

    stages = cfg.report.stages
    outs = []
    for i in range(len(stages)):
        exec_stages = stages[:i]
        pipeline_out = pipeline(
            data=data,
            data_mode=DataMode.OFFLINE,
            reset_temporal_state=False,
            stages=exec_stages
        )
        outs.append(pipeline_out.df)

    generate_report(cfg.report, outs, stages)


if __name__ == "__main__":
    main()
