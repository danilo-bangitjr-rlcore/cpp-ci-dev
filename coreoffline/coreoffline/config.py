from datetime import datetime, timedelta
from pathlib import Path

from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import StageCode
from lib_config.config import MISSING, computed, config, post_processor
from pydantic import Field


@config()
class OfflineTrainingConfig:
    offline_steps: int = 0
    offline_start_time: datetime | None = None
    offline_end_time: datetime | None = None
    eval_periods: list[tuple[datetime, datetime]] | None = None
    pipeline_batch_duration: timedelta = timedelta(days=7)
    update_agent_during_offline_recs: bool = False
    remove_eval_from_train: bool = True
    test_split: float = 0.0

    @post_processor
    def _validate(self, cfg: 'OfflineMainConfig'):
        if isinstance(self.offline_start_time, datetime) and isinstance(self.offline_end_time, datetime):
            assert (
                self.offline_start_time < self.offline_end_time
            ), "Offline training start timestamp must come before the offline training end timestamp."

        if self.eval_periods is not None:
            for eval_period in self.eval_periods:
                assert len(eval_period) == 2, "Eval periods must be defined as a list with length 2."
                start = eval_period[0]
                end = eval_period[1]
                assert start < end, "Eval start must precede eval end."
            # assert isinstance(self.eval_periods, list[tuple[datetime, datetime]])


@config()
class ReportConfig:
    output_dir: Path = Path('outputs/report')
    stages: list[StageCode] = Field(default_factory=lambda: [StageCode.INIT])
    tags_to_exclude: list = Field(default_factory=list)  # tags to exclude from analysis

    # for stat table
    stat_table_enabled: bool = True

    # for cross correlation
    # options for cross_corr_tags:
    # 1. list of tag names -> will find cross correlation for all pairs of tags in this list
    # 2. list of list[str] -> will find cross correlation only for these pairs in each list.
    # 3. None -> will find cross correlation for ALL pairs of tags
    cross_corr_enabled: bool = True
    cross_corr_tags: list[str] | list[list[str]] | None = Field(default_factory=list)
    cross_corr_max_lag: int = 100

    # for histograms
    hist_enabled: bool = True
    hist_show_mean: bool = True
    hist_percentiles: list[float] = Field(default_factory=lambda: [0.1, 0.9])
    hist_num_bins: int = 30

    # for transition statistics
    transition_percentiles: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    contiguous_time_threshold: timedelta = MISSING  # max time gap to consider transitions contiguous

    # for goal violations
    violation_period_percentiles: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])

    @computed('contiguous_time_threshold')
    @classmethod
    def _contiguous_time_threshold(cls, cfg: 'OfflineMainConfig'):
        return cfg.interaction.obs_period


@config()
class OfflineMainConfig(MainConfig):
    offline: OfflineTrainingConfig = Field(default_factory=OfflineTrainingConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
