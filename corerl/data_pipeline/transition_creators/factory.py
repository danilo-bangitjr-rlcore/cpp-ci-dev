from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transition_creators.base import BaseTransitionCreator, BaseTransitionCreatorConfig, \
    transition_creator_group

from corerl.data_pipeline.transition_creators.anytime import AnytimeTransitionCreatorConfig
from corerl.data_pipeline.transition_creators.dummy import DummyTransitionCreatorConfig


TransitionCreatorConfig = (
    AnytimeTransitionCreatorConfig
    | DummyTransitionCreatorConfig
)

def init_transition_creator(cfg: BaseTransitionCreatorConfig, tags: list[TagConfig]) -> BaseTransitionCreator:
    return transition_creator_group.dispatch(cfg, tags)
