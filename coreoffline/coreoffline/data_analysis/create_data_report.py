import logging
import random
from datetime import datetime

import numpy as np
import pandas as pd
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics.factory import create_metrics_writer
from corerl.messages.event_bus import DummyEventBus
from corerl.state import AppState
from corerl.utils.pandas import split_dataframe_into_chunks
from lib_agent.buffer.datatypes import DataMode
from lib_config.loader import load_config

from coreoffline.config import OfflineMainConfig
from coreoffline.data_analysis.data_report import generate_report
from coreoffline.data_loading import load_data_chunks, load_entire_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class StageDataCapture:
    """Captures pipeline dataframe state after each specified stage"""
    def __init__(self, pipeline: Pipeline):
        self.captured_data: dict[StageCode, list[pd.DataFrame]] = {}
        stages = pipeline.default_stages

        # Initialize empty lists for each stage
        for stage in stages:
            self.captured_data[stage] = []

        for stage in stages:
            pipeline.register_hook(
                data_modes=DataMode.OFFLINE,
                stages=stage,
                f=self.create_capture_hook(stage),
                order='post',
        )

    def create_capture_hook(self, stage: StageCode):
        """Returns a hook function that captures dataframe state after specified stage"""
        def hook(pf: PipelineFrame) -> None:
            self.captured_data[stage].append(pf.data.copy())

        return hook

    def get_concatenated_data(self, stage: StageCode) -> pd.DataFrame:
        """Returns concatenated dataframe for the specified stage"""
        if not self.captured_data[stage]:
            return pd.DataFrame()
        return pd.concat(self.captured_data[stage])


@load_config(OfflineMainConfig)
def main(cfg: OfflineMainConfig):
    """
    Assuming offline data has already been written to TimescaleDB
    """
    # set the random seeds

    seed = cfg.seed
    np.random.seed(seed)
    random.seed(seed)
    start_time = datetime.now()

    app_state = AppState(
        cfg,
        evals=EvalsTable(cfg.evals),
        metrics=create_metrics_writer(cfg.metrics),
        event_bus=DummyEventBus(),
    )
    assert isinstance(app_state.cfg, OfflineMainConfig)  # for typing

    pipeline = Pipeline(app_state, cfg.pipeline)
    log.info("Loading dataset...")

    data = load_entire_dataset(cfg)
    data_chunks = split_dataframe_into_chunks(data, 10_000)

    log.info("Loaded Dataset")
    capture = StageDataCapture(pipeline)

    # Single pipeline execution through all stages
    log.info("Running pipeline with stage capture hooks...")

    exclude_periods = cfg.offline_training.eval_periods if cfg.offline_training.remove_eval_from_train else None
    data_chunks = load_data_chunks(
        cfg=app_state.cfg,
        start_time=cfg.offline_training.offline_start_time,
        end_time=cfg.offline_training.offline_end_time,
        exclude_periods=exclude_periods,
    )

    transitions = []
    for chunk in data_chunks:
        pr = pipeline(
            data=chunk,
            data_mode=DataMode.OFFLINE,
            reset_temporal_state=False,
        )
        if pr.transitions is not None:
            transitions += pr.transitions

    end_time = datetime.now()

    # Extract captured dataframes
    data = []
    for stage in cfg.report.stages:
        data.append(capture.get_concatenated_data(stage))

    log.info("Generating report from captured stage data...")
    generate_report(
        cfg.report,
        data,
        cfg.report.stages,
        app_state,
        start_time,
        end_time,
        transitions,
    )


if __name__ == "__main__":
    main()
