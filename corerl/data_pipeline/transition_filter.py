from corerl.utils.torch import tensor_allclose
from dataclasses import dataclass, field

from corerl.data_pipeline.datatypes import PipelineFrame


@dataclass
class TransitionFilterConfig:
    # doing it this way instead of list so that we can specify some default strings when we agree on them
    filters: list[str] = field(default_factory=lambda: [])


class TransitionFilter:
    def __init__(self, cfg: TransitionFilterConfig):
        self.filter_names = cfg.filters

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        for filter_name in self.filter_names:
            pf.transitions = call_filter(pf.transitions, filter_name)

        return pf


def call_filter(transitions, filter_name):
    if filter_name == 'only_dp_transition':
        transition_filter = only_dp_transition
    elif filter_name == 'only_no_action_change':
        transition_filter = only_no_action_change
    elif filter_name == 'only_post_dp_transition':
        transition_filter = only_post_dp_transition
    else:
        raise NotImplementedError(f"Invalid transition filter name {filter_name}")

    results = [transition_filter(transition) for transition in transitions]
    filtered = [transition for transition, keep in zip(transitions, results) if keep]
    return filtered


def only_dp_transition(transition):
    if transition.prior.dp and transition.post.dp:
        return True
    else:
        return False


def only_post_dp_transition(transition):
    if transition.post.dp:
        return True
    else:
        return False


def only_no_action_change(transition):
    action = transition.steps[1].action
    for i in range(1, len(transition.steps)):
        if not tensor_allclose(transition.steps[i].action, action):
            return False
    return True
