import logging
from datetime import datetime

from lib_agent.buffer.datatypes import DataMode
from lib_config.loader import load_config
from lib_progress.tracker import track

from coreoffline.utils.config import OfflineMainConfig
from coreoffline.utils.data_analysis.transition_report import generate_report
from coreoffline.utils.data_loading import load_data_chunks
from coreoffline.utils.setup import create_standard_setup

log = logging.getLogger(__name__)


@load_config(OfflineMainConfig)
def main(cfg: OfflineMainConfig):
    """
    Assuming offline data has already been written to TimescaleDB
    """
    app_state, pipeline = create_standard_setup(cfg)

    # Single pipeline execution through all stages
    log.info("Running pipeline with stage capture hooks...")

    exclude_periods = cfg.offline_training.eval_periods if cfg.offline_training.remove_eval_from_train else None
    data_chunks, num_chunks = load_data_chunks(
        cfg=app_state.cfg,
        start_time=cfg.offline_training.offline_start_time,
        end_time=cfg.offline_training.offline_end_time,
        exclude_periods=exclude_periods,
    )
    start_time = datetime.now()

    transitions = []
    for chunk in track(
        data_chunks,
        desc="Running data through pipeline",
        total=num_chunks,
        update_interval=5,
    ):
        pr = pipeline(
            data=chunk,
            data_mode=DataMode.OFFLINE,
            reset_temporal_state=False,
        )
        if pr.transitions is not None:
            transitions += pr.transitions

    end_time = datetime.now()

    log.info("Generating report from captured stage data...")
    generate_report(
        cfg.report,
        app_state,
        start_time,
        end_time,
        transitions,
    )

    log.info("=" * 80)
    log.info("Transition report generation complete!")
    log.info(f"📁 Artifacts saved to: {cfg.report.output_dir.resolve()}")
    log.info("📋 Reports: transition_statistics (txt/csv)")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
