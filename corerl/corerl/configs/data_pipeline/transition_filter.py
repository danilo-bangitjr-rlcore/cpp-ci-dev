
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lib_config.config import config, list_, post_processor

if TYPE_CHECKING:
    from corerl.config import MainConfig

type TransitionFilterType = (
    Literal['only_dp', 'only_no_action_change', 'only_post_dp', 'no_nan', 'only_pre_dp_or_ac']
)


@config()
class TransitionFilterConfig:
    """
    Kind: internal

    A list of filters on transitions to recover regular rl transitions,
    anytime transitions, all-the-time transitions, etc.
    """
    filters: list[TransitionFilterType] = list_(['no_nan'])

    @post_processor
    def _validate_compatible_filters(self, cfg: MainConfig):
        if "only_dp" in self.filters:
            assert (
                'only_post_dp' not in self.filters
            ), "'only_dp' and 'only_post_dp' are inconsistent transition filters."

            assert (
                'only_pre_dp_or_ac' not in self.filters
            ), "'only_dp' and 'only_pre_dp_or_ac' are inconsistent transition filters."
