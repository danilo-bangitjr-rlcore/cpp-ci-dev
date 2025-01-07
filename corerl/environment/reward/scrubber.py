import logging
from typing import Any
from warnings import warn

import numpy as np
import torch
import pandas as pd

from corerl.configs.config import MISSING, config
from corerl.environment.reward.base import BaseReward

logger = logging.getLogger(__name__)


def contains_nan(arr: np.ndarray | pd.DataFrame | torch.Tensor) -> bool:
    if isinstance(arr, np.ndarray):
        return bool(np.isnan(arr).any())
    elif isinstance(arr, pd.DataFrame):
        return bool(arr.isnull().values.any())
    elif isinstance(arr, torch.Tensor):
        return bool(torch.isnan(arr).any())

@config()
class ScrubberRewardConfig:
    reward_tags: Any = MISSING
    steps_per_interaction: Any = MISSING
    only_dp_transitions: bool = False
    type: str = 'mean_efficiency_cost'

class ScrubberReward(BaseReward):
    """
    Reward for scrubber4 EPCOR data
    """
    def __init__(self, cfg: Any):
        self.reward_tags = cfg.reward_tags
        self._steps_per_interaction = cfg.steps_per_interaction
        self._only_dp_transitions = cfg.only_dp_transitions
        if self._only_dp_transitions:
            self._t = 0

        self._type = cfg.type
        if self._type in ("final_efficiency", "efficiency"):
            self._reward_fn = self._efficiency_reward_final
        elif self._type == "mean_efficiency":
            self._reward_fn = self._efficiency_reward_step
        elif self._type == "mean_cost":
            self._reward_fn = lambda x: self._compute_eff_cost(x)[1].mean()
        elif self._type == "mean_efficiency_cost":
            self._reward_fn = self._efficiency_cost_reward_step
        elif self._type == "learned":
            # For backwards compatibility, we will keep the learned reward for
            # now
            self._reward_fn = self._learned_reward
        else:
            raise NotImplementedError(f"unknown reward type {self._type}")

    @classmethod
    def compute_chem_cost(
        cls, orp_pumpspeed: np.ndarray, ph_pumpspeed: np.ndarray
    ) -> np.ndarray:
        orp_cost = 1.355 * orp_pumpspeed
        ph_cost = 0.5455 * ph_pumpspeed

        total_cost = orp_cost + ph_cost
        return total_cost

    @classmethod
    def _compute_eff_cost(
        cls, df_rows: pd.DataFrame, eff_tag: str = "AI0879C", orp_tag: str = "AIC3731_OUT", ph_tag: str = "AIC3730_OUT"
    ):
        """
        Compute and return the efficiency and cost given some rows from the
        Epcor dataset.
        """
        # logger.debug("_compute_eff_cost")
        # logger.debug(f"{df_rows=}")
        eff = df_rows[eff_tag]
        if eff_tag == "AI0879C":
            # this efficiency tag is in range 0-100
            eff /= 100.0

        if isinstance(eff, np.floating):
            mean_eff = eff
        else:
            mean_eff = df_rows[eff_tag].to_numpy().mean()

        orp_use = np.array(df_rows[orp_tag])
        ph_use = np.array(df_rows[ph_tag])
        chem_cost = cls.compute_chem_cost(orp_pumpspeed=orp_use, ph_pumpspeed=ph_use)

        return mean_eff, chem_cost

    def _learned_reward(self, df_rows: Any):
        warn("learned reward is deprecated and will be removed in the future",
             stacklevel=1)

        eff, cost = ScrubberReward._compute_eff_cost(df_rows)

        assert isinstance(eff, (list, pd.Series, np.ndarray))
        if len(eff) == 0:
            r = 0.0
        else:
            r = (eff - 0.56234210729599 * cost).mean()

        return r

    def _efficiency_reward_step(self, df_rows: Any):
        """
        This function will return as reward the mean efficiency across
        `only_dp_transitions`, which should be the mean efficiency over 1 hour
        for the Epcor data.
        """
        return df_rows["Efficiency"].to_numpy().mean()

    def _efficiency_cost_reward_step(self, df_rows: pd.DataFrame):
        """
        This function implements the following reward:

                        ⎧ e / 1.90                      if e < 0.95
            r(e, c) =   ⎨
                        ⎩ (-0.95 c + 1.90) / 1.90       otherwise


        where `e` is the efficiency, `c` is the cost normalized by
        `(c_min, c_max) = (0, 200)`, and `r(e, c) ∈ [0, 1] ∀e,c∈[0, 1]`.

        For `e0 < 0.95` and `e1 >= 0.95`:

            r(e0, c) < r(e1, c)     ∀c>=0

        Hence, reward is always maximized by increasing efficiency. Further
        for `c0 < c1`:

            r(e, c0) >= r(e, c1)    ∀e>=0

        So that reward is maximized by decreasing cost. Note that the above
        inequality is strict for `e >= 0.95`.
        """
        assert not contains_nan(df_rows)
        # logger.debug("_efficiency_cost_reward_step")
        eff, cost = ScrubberReward._compute_eff_cost(df_rows)

        if eff < 0.95:
            r = eff
        else:
            # Normalize cost
            #
            # As of July 29, 2024, (cost_min, cost_max) ≅ (0, 185) in the
            # dataset provided by Epcor.
            cost_min, cost_max = 0, 200
            c = (cost - cost_min) / (cost_max - cost_min)

            r = -0.95 * c + 1.90

        # Normalize reward
        r_min = 0
        r_max = 1.90
        return float(np.mean((r - r_min) / (r_max - r_min)))

    def _efficiency_reward_final(self, df_rows: Any) -> float:
        """
        This function will return as reward the final efficiency after
        `only_dp_transitions`, which should be the final efficiency after 1
        hour for the Epcor data.
        """
        r = self._efficiency_reward_step(df_rows)

        if self._only_dp_transitions:
            # In this case, the transition creator will take the average reward
            # over each sub-transition of the anytime transition, so we need to
            # scale the final reward by self._steps_per_interaction.
            self._t += 1
            if not (self._t % self._steps_per_interaction):
                return r * self._steps_per_interaction
            else:
                return 0.0
        else:
            return r

    def __call__(self, obs: Any, *args: Any, **kwargs: Any) -> float:
        df_rows = kwargs["df_rows"]
        assert not contains_nan(df_rows)
        reward = self._reward_fn(df_rows)
        return reward
