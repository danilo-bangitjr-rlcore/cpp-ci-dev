from corerl.data_pipeline.transition_creators.base import BaseTransitionCreator, BaseTransitionCreatorConfig, \
    transition_creator_group


def init_transition_creator(cfg: BaseTransitionCreatorConfig) -> BaseTransitionCreator:
    return transition_creator_group.dispatch(cfg)
