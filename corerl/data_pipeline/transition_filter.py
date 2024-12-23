from typing import Literal, assert_never
from corerl.utils.torch import tensor_allclose

from corerl.configs.config import config, list_
from corerl.data_pipeline.datatypes import PipelineFrame


type TransitionFilterType = (
    Literal['only_dp']
    | Literal['only_no_action_change']
    | Literal['only_post_dp']
)


@config()
class TransitionFilterConfig:
    filters: list[TransitionFilterType] = list_()


class TransitionFilter:
    def __init__(self, cfg: TransitionFilterConfig):
        self.filter_names = cfg.filters

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        if pf.transitions is None:
            return pf

        for filter_name in self.filter_names:
            pf.transitions = call_filter(pf.transitions, filter_name)

        return pf

def call_filter(transitions, filter_name):
    if filter_name == 'only_dp':
        transition_filter = only_dp
    elif filter_name == 'only_no_action_change':
        transition_filter = only_no_action_change
    elif filter_name == 'only_post_dp':
        transition_filter = only_post_dp
    else:
        assert_never(filter_name)

    results = [transition_filter(transition) for transition in transitions]
    filtered = [transition for transition, keep in zip(transitions, results, strict=True) if keep]
    return filtered


def only_dp(transition):
    return transition.prior.dp and transition.post.dp


def only_post_dp(transition):
    return transition.post.dp


def only_no_action_change(transition):
    action = transition.steps[1].action
    for i in range(1, len(transition.steps)):
        if not tensor_allclose(transition.steps[i].action, action):
            return False
    return True
