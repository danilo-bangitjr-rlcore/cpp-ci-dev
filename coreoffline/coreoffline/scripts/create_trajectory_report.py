import logging
from datetime import datetime

from lib_agent.buffer.datatypes import DataMode
from lib_config.loader import load_config
from lib_progress.tracker import track

from coreoffline.utils.config import OfflineMainConfig
from coreoffline.utils.data_analysis.trajectory_report import generate_report
from coreoffline.utils.data_loading import load_data_chunks
from coreoffline.utils.setup import create_standard_setup

log = logging.getLogger(__name__)


@load_config(OfflineMainConfig)
def main(cfg: OfflineMainConfig):
    """
    Assuming offline data has already been written to TimescaleDB
    """
    log.info("=" * 80)
    log.info("Starting trajectory report generation")
    log.info("=" * 80)

    app_state, pipeline = create_standard_setup(cfg)

    # Single pipeline execution through all stages
    start_time = cfg.offline_training.offline_start_time
    end_time = cfg.offline_training.offline_end_time
    log.info(f"Processing data from {start_time} to {end_time}")
    log.info("Running pipeline to generate trajectorys...")

    exclude_periods = cfg.offline_training.eval_periods if cfg.offline_training.remove_eval_from_train else None
    data_chunks, num_chunks = load_data_chunks(
        cfg=app_state.cfg,
        start_time=start_time,
        end_time=end_time,
        exclude_periods=exclude_periods,
    )
    report_gen_start = datetime.now()

    trajectories = []
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
        if pr.trajectories is not None:
            trajectories += pr.trajectories

    report_gen_end = datetime.now()

    log.info(f"Generated {len(trajectories)} trajectories")
    log.info("Generating trajectory report...")
    generate_report(
        cfg.report,
        app_state,
        report_gen_start,
        report_gen_end,
        trajectories,
    )

    log.info("=" * 80)
    log.info("Trajectory report generation complete!")
    log.info(f"📁 Artifacts saved to: {cfg.report.output_dir.resolve()}")
    log.info("📋 Reports: trajectory_statistics (txt/csv)")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
