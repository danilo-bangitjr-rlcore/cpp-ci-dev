from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING

from pydantic import Field

from corerl.configs.config import MISSING, config, list_, sanitizer
from corerl.data_pipeline.imputers.per_tag.factory import ImputerConfig
from corerl.data_pipeline.oddity_filters.factory import OddityFilterConfig
from corerl.data_pipeline.oddity_filters.identity import IdentityFilterConfig
from corerl.data_pipeline.transforms import NormalizerConfig, NullConfig, TransformConfig
from corerl.utils.list import find_instance
from corerl.utils.maybe import Maybe

if TYPE_CHECKING:
    from corerl.config import MainConfig

Bounds = tuple[float | None, float | None]

class Agg(StrEnum):
    avg = auto()
    last = auto()
    bool_or = auto()

@config()
class TagConfig:
    """This is the configuration for our OPC tags. A tag is roughly equivalent to a variable that is sent and received
    through an OPC server. It is typically a single primative datatype (e.g. float).

    From `PR#335 Discussion <https://github.com/rlcoretech/core-rl/pull/335#discussion_r1898067439>`_ a tag *may*
    simultaneously represent both an observation, a reward, and possibly also an action.
    """

    # tag metadata
    name: str = MISSING
    agg: Agg = Agg.avg
    node_identifier: str | int | None = None
    is_meta: bool = False
    is_endogenous: bool = True

    # tag zones
    operating_range: Bounds | None = None
    red_bounds: Bounds | None = None
    yellow_bounds: Bounds | None = None

    # per-tag pipeline configuration
    outlier: OddityFilterConfig = Field(default_factory=IdentityFilterConfig)
    imputer: ImputerConfig | None = None

    # per-tag constructors
    preprocess: list[TransformConfig] = list_([NormalizerConfig()])
    reward_constructor: list[TransformConfig] = list_([NullConfig()])
    action_constructor: list[TransformConfig] | None = None
    state_constructor: list[TransformConfig] | None = None
    filter: list[TransformConfig] | None = None

    @sanitizer
    def _default_normalize_preprocessor(self, cfg: MainConfig):
        # each bound type is fully optional
        # prefer to use red zone, fallback to black zone then yellow
        # finally fallback to learning online if no other options
        # available
        lo = (
            Maybe[float](self.red_bounds and self.red_bounds[0])
            .otherwise(lambda: self.operating_range and self.operating_range[0])
            .otherwise(lambda: self.yellow_bounds and self.yellow_bounds[0])
        )

        hi = (
            Maybe[float](self.red_bounds and self.red_bounds[1])
            .otherwise(lambda: self.operating_range and self.operating_range[1])
            .otherwise(lambda: self.yellow_bounds and self.yellow_bounds[1])
        )

        # although each constructor type may _also_ have a normalizer
        # only automatically set the preprocessor normalizer bounds
        norm_cfg = find_instance(NormalizerConfig, self.preprocess)
        if norm_cfg is None:
            return

        norm_cfg.min = (
            Maybe(norm_cfg.min)
            .flat_otherwise(lambda: lo)
            .unwrap()
        )

        norm_cfg.max = (
            Maybe(norm_cfg.max)
            .flat_otherwise(lambda: hi)
            .unwrap()
        )

        if norm_cfg.min is None or norm_cfg.max is None:
            norm_cfg.from_data = True
