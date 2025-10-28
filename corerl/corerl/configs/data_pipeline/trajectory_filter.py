
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lib_config.config import config, list_, post_processor

if TYPE_CHECKING:
    from corerl.config import MainConfig

type TrajectoryFilterType = (
    Literal['only_dp', 'only_no_action_change', 'only_post_dp', 'no_nan', 'only_pre_dp_or_ac']
)


@config()
class TrajectoryFilterConfig:
    """
    Kind: internal

    A list of filters on trajectories to recover regular rl trajectories,
    anytime trajectories, all-the-time trajectories, etc.
    """
    filters: list[TrajectoryFilterType] = list_(['no_nan'])

    @post_processor
    def _validate_compatible_filters(self, cfg: MainConfig):
        if "only_dp" in self.filters:
            assert (
                'only_post_dp' not in self.filters
            ), "'only_dp' and 'only_post_dp' are inconsistent trajectory filters."

            assert (
                'only_pre_dp_or_ac' not in self.filters
            ), "'only_dp' and 'only_pre_dp_or_ac' are inconsistent trajectory filters."
