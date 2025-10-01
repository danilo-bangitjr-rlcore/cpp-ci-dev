import logging
from datetime import datetime

from lib_agent.buffer.datatypes import DataMode
from lib_config.loader import load_config

from coreoffline.config import OfflineMainConfig
from coreoffline.core.pipeline_utils import StageDataCapture
from coreoffline.core.setup import create_standard_setup
from coreoffline.data_analysis.data_report import generate_report
from coreoffline.data_loading import load_data_chunks

log = logging.getLogger(__name__)


@load_config(OfflineMainConfig)
def main(cfg: OfflineMainConfig):
    """
    Assuming offline data has already been written to TimescaleDB
    """
    app_state, pipeline = create_standard_setup(cfg)
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
    start_time = datetime.now()

    for chunk in data_chunks:
        pipeline(
            data=chunk,
            data_mode=DataMode.OFFLINE,
            reset_temporal_state=False,
        )

    end_time = datetime.now()

    # Extract captured dataframes
    data = [capture.get_concatenated_data(stage) for stage in cfg.report.stages]

    log.info("Generating report from captured stage data...")
    generate_report(
        cfg.report,
        data,
        cfg.report.stages,
        app_state,
        start_time,
        end_time,
    )


if __name__ == "__main__":
    main()
