
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from lib_config.config import config, list_, post_processor
from pydantic import Field

from corerl.configs.data_pipeline.imputers.base import BaseImputerStageConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config(frozen=True)
class TrainingConfig:
    init_train_steps = 100
    batch_size: int = 64
    stepsize: float = 1e-4
    err_tolerance: float = 1e-4
    max_update_steps: int = 100
    training_missing_perc: float = 0.25


@config()
class MaskedAEConfig(BaseImputerStageConfig):
    name: Literal["masked-ae"] = "masked-ae"
    horizon: int = 10
    trace_values: list[float] = list_([0.5, 0.9, 0.95])
    buffer_size: int = 200_000

    fill_val: float = 0.0
    prop_missing_tol: float = np.nan
    train_cfg: TrainingConfig = Field(default_factory=TrainingConfig)

    @post_processor
    def _set_missing_tol(self, cfg: MainConfig):
        """
        If no missing tol is explicitly set, use the training_missing_perc
        """
        if not np.isnan(self.prop_missing_tol):
            return
        self.prop_missing_tol = self.train_cfg.training_missing_perc
