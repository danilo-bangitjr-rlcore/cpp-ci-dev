
from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Literal

from lib_config.config import MISSING, computed, config, post_processor

from corerl.configs.data_pipeline.transforms.base import BaseTransformConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class DeltaConfig(BaseTransformConfig):
    name: Literal["delta"] = "delta"
    time_thresh: timedelta = MISSING
    obs_period: timedelta = MISSING

    @computed("time_thresh")
    @classmethod
    def _time_thresh(cls, cfg: MainConfig):
        return cfg.interaction.obs_period

    @computed("obs_period")
    @classmethod
    def _obs_period(cls, cfg: MainConfig):
        return cfg.interaction.obs_period

    @post_processor
    def _validate_obs_period_alignment(self, cfg: MainConfig):
        assert (
            self.time_thresh >= cfg.interaction.obs_period
        ), (
            "New sensor readings are only observed every obs_period, therefore at least obs_period must elapse "
            "before deltas can be computed. DeltaConfig.time_thresh must be greater or equal to obs_period."
        )
