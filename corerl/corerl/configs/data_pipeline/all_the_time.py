
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from lib_config.config import MISSING, computed, config, post_processor

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class AllTheTimeTCConfig:
    """
    Kind: internal

    Configures the length of n-step trajectories produced by
    the pipeline.
    """

    name: str = "all-the-time"
    min_n_step: int = 1
    max_n_step: int = MISSING
    gamma: float = MISSING
    normalize_return: bool = MISSING

    @computed('max_n_step')
    @classmethod
    def _max_n_step(cls, cfg: MainConfig):
        ap_sec = cfg.interaction.action_period.total_seconds()
        obs_sec = cfg.interaction.obs_period.total_seconds()

        steps_per_decision = int(ap_sec / obs_sec)
        assert np.isclose(steps_per_decision, ap_sec / obs_sec), \
            "Action period must be a multiple of obs period"

        return steps_per_decision

    @computed('gamma')
    @classmethod
    def _gamma(cls, cfg: MainConfig):
        return cfg.agent.gamma

    @computed('normalize_return')
    @classmethod
    def _normalize_return(cls, cfg: MainConfig):
        return cfg.feature_flags.normalize_return

    @post_processor
    def _validate_on_policy_trajectory_creation(self, cfg: MainConfig):
        assert (
            self.min_n_step >= 1
        ), "n-step trajectories must span at least one obs_period. " \
           "Therefore, AllTheTimeTCConfig.min_n_step must be greater or equal to 1."

        assert (
            self.max_n_step >= self.min_n_step
        ), "AllTheTimeTCConfig.max_n_step must be greater or equal to AllTheTimeTCConfig.min_n_step"

        assert (
            self.max_n_step <= (cfg.interaction.action_period / cfg.interaction.obs_period)
        ), "AllTheTimeTCConfig.max_n_step must span less than or equal to the number of obs_periods in action_period"
