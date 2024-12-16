from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transition_creators.base import BaseTransitionCreator, BaseTransitionCreatorConfig, \
    transition_creator_group

import corerl.data_pipeline.transition_creators.anytime # noqa: F401

def init_transition_creator(cfg: BaseTransitionCreatorConfig, tags: list[TagConfig]) -> BaseTransitionCreator:
    return transition_creator_group.dispatch(cfg, tags)
