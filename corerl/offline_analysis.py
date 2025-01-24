import logging
import random

import numpy as np
import torch

import corerl.main_utils as utils
from corerl.config import MainConfig
from corerl.configs.loader import load_config
from corerl.data_pipeline.datatypes import CallerCode, StageCode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.data_report import generate_report
from corerl.offline.utils import load_entire_dataset
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

    pipeline = Pipeline(cfg.pipeline)
    log.info("loading dataset...")
    data = load_entire_dataset(cfg)

    stages = [StageCode.INIT, StageCode.BOUNDS]
    outs = []
    for i in range(len(stages)):
        exec_stages = stages[:i]
        pipeline_out = pipeline(
            data=data,
            caller_code=CallerCode.OFFLINE,
            reset_temporal_state=False,
            stages=exec_stages
        )
        outs.append(pipeline_out.df)

    generate_report(cfg.report, outs, stages)



if __name__ == "__main__":
    main()
