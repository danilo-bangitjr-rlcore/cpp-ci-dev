
from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from lib_config.config import MISSING, computed, config

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class CountdownConfig:
    """
    Kind: internal

    Enables a countdown "virtual tag" as a feature in the constructed
    state. The countdown is a function of both action_period
    and obs_period, signaling the number of steps until the next
    decision point.
    """

    kind: str = 'no_countdown'
    normalize: bool = True

    action_period: timedelta = MISSING
    obs_period: timedelta = MISSING
    @computed('action_period')
    @classmethod
    def _action_period(cls, cfg: MainConfig):
        return cfg.interaction.action_period

    @computed('obs_period')
    @classmethod
    def _obs_period(cls, cfg: MainConfig):
        return cfg.interaction.obs_period
