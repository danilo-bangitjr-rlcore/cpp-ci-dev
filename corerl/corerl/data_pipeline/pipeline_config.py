from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from lib_config.config import MISSING, computed, config, list_, post_processor
from lib_defs.config_defs.tag_config import TagType
from pydantic import Field

from corerl.configs.tags.components.bounds import BoundType, init_bounds_info
from corerl.configs.tags.components.opc import Agg
from corerl.configs.tags.tag_config import BasicTagConfig, TagConfig, in_taglist
from corerl.data_pipeline.all_the_time import AllTheTimeTCConfig
from corerl.data_pipeline.constructors.sc import SCConfig
from corerl.data_pipeline.imputers.auto_encoder import MaskedAEConfig
from corerl.data_pipeline.imputers.factory import ImputerStageConfig
from corerl.data_pipeline.imputers.imputer_stage import PerTagImputerConfig
from corerl.data_pipeline.oddity_filters.config import GlobalOddityFilterConfig
from corerl.data_pipeline.oddity_filters.identity import IdentityFilterConfig
from corerl.data_pipeline.transforms import NukeConfig, register_dispatchers
from corerl.data_pipeline.transition_filter import TransitionFilterConfig
from corerl.data_pipeline.virtual.deltaize_tags import DeltaStageConfig
from corerl.environment.reward.config import RewardConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig

logger = logging.getLogger(__name__)
register_dispatchers()


@config()
class PipelineConfig:
    tags: list[TagConfig] = list_()
    max_data_gap: timedelta = MISSING

    # stage-wide configs
    delta: DeltaStageConfig = Field(default_factory=DeltaStageConfig)
    imputer: ImputerStageConfig = Field(default_factory=MaskedAEConfig)
    oddity_filter: GlobalOddityFilterConfig = Field(default_factory=GlobalOddityFilterConfig)
    state_constructor: SCConfig = Field(default_factory=SCConfig)
    transition_creator: AllTheTimeTCConfig = Field(default_factory=AllTheTimeTCConfig)
    transition_filter: TransitionFilterConfig = Field(default_factory=TransitionFilterConfig)
    reward: RewardConfig | None = None

    @computed('max_data_gap')
    @classmethod
    def _max_data_gap(cls, cfg: MainConfig):
        return 2 * cfg.interaction.obs_period

    @post_processor
    def _cascade_dependencies(self, cfg: MainConfig):
        tags = {tag.name for tag in self.tags}
        for tag in self.tags:
            if tag.type != TagType.ai_setpoint or tag.cascade is None:
                continue
            for dep in [tag.cascade.op_sp, tag.cascade.ai_sp]:
                if in_taglist(dep, self.tags): continue
                new_tag = BasicTagConfig(
                    name=dep,
                    agg=Agg.last,
                    operating_range=tag.operating_range,
                    expected_range=tag.expected_range,
                    preprocess=[],
                    state_constructor=[NukeConfig()],
                )
                if new_tag.operating_range is not None:
                    new_tag.operating_bounds_info = init_bounds_info(
                        cfg=new_tag,
                        bounds=new_tag.operating_range,
                        bound_type=BoundType.operating_range,
                        known_tags=tags,
                    )
                if new_tag.expected_range is not None:
                    new_tag.expected_bounds_info = init_bounds_info(
                        cfg=new_tag,
                        bounds=new_tag.expected_range,
                        bound_type=BoundType.expected_range,
                        known_tags=tags,
                    )
                self.tags.append(new_tag)

            if in_taglist(tag.cascade.mode, self.tags): continue
            min_op = min(tag.cascade.op_mode_val, tag.cascade.ai_mode_val)
            max_op = max(tag.cascade.op_mode_val, tag.cascade.ai_mode_val)
            mode_tag = BasicTagConfig(
                name=tag.cascade.mode,
                agg=Agg.bool_or if tag.cascade.mode_is_bool else Agg.last,
                operating_range=(min_op, max_op),
                preprocess=[],
                state_constructor=[NukeConfig()],
                outlier=[IdentityFilterConfig()],
            )
            if mode_tag.operating_range is not None:
                mode_tag.operating_bounds_info = init_bounds_info(
                    cfg=mode_tag,
                    bounds=mode_tag.operating_range,
                    bound_type=BoundType.operating_range,
                    known_tags=tags,
                )
            self.tags.append(mode_tag)

    @post_processor
    def _default_imputers(self, cfg: MainConfig):
        if not isinstance(self.imputer, PerTagImputerConfig):
            return

        for tag in self.tags:
            if tag.imputer is not None:
                continue

            tag.imputer = self.imputer.default

    @post_processor
    def _validate_obs_period_alignment(self, cfg: MainConfig):
        assert (
            self.max_data_gap >= cfg.interaction.obs_period
        ), (
            "Temporal state can only become stale after at least obs_period has elapsed. "
            "max_data_gap must be greater or equal to obs_period."
        )

    @post_processor
    def _validate_tags(self, cfg: MainConfig):
        seen = set[str]()
        for tag in self.tags:
            if tag.name in seen:
                raise ValueError(f"Duplicate tag name found: {tag.name}")

            seen.add(tag.name)
