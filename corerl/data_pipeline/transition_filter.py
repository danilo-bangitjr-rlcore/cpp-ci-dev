import math
from collections.abc import Iterable
from typing import Literal, assert_never

import torch

from corerl.configs.config import config, list_
from corerl.data_pipeline.datatypes import PipelineFrame, Transition

type TransitionFilterType = (
    Literal['only_dp']
    | Literal['only_no_action_change']
    | Literal['only_post_dp']
    | Literal['no_nan']
    | Literal['only_pre_dp_or_ac']
)


@config()
class TransitionFilterConfig:
    filters: list[TransitionFilterType] = list_(['no_nan'])


class TransitionFilter:
    def __init__(self, cfg: TransitionFilterConfig):
        self.filter_names = cfg.filters

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        if pf.transitions is None:
            return pf

        for filter_name in self.filter_names:
            pf.transitions = call_filter(pf.transitions, filter_name)

        return pf


def call_filter(transitions: Iterable[Transition], filter_name: TransitionFilterType):
    if filter_name == 'only_dp':
        transition_filter = only_dp
    elif filter_name == 'only_no_action_change':
        transition_filter = only_no_action_change
    elif filter_name == 'only_post_dp':
        transition_filter = only_post_dp
    elif filter_name == 'no_nan':
        transition_filter = no_nan
    elif filter_name == 'only_pre_dp_or_ac':
        transition_filter = only_pre_dp_or_ac
    else:
        assert_never(filter_name)

    return list(filter(transition_filter, transitions))


def only_dp(transition: Transition):
    return transition.prior.dp and transition.post.dp

def only_pre_dp_or_ac(transition: Transition):
    return transition.prior.dp or transition.steps[1].ac

def only_post_dp(transition: Transition):
    return transition.post.dp


def only_no_action_change(transition: Transition):
    """
    The initial action change typically occurs on index 1:
        The agent takes action transition.steps[1].action in response
        to the state from transition.steps[0].state.
    This function checks for action changes after index 1.
    """
    i = 2 # check for action changes after initial action
    while i < len(transition.steps):
        # trust the countdown creator
        # NOTE: otherwise, we need to deal with delta actions correctly
        if transition.steps[i].ac:
            return False
        i += 1

    return True


def has_nan(obj: object):
    for _, value in vars(obj).items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                return True
        elif isinstance(value, float) and math.isnan(value):
            return True
    return False


def no_nan(transition: Transition):
    """Checks to see if there are any nans in the transition. Ignores the reward and action
    on the first step in the transition, as it is valid for this to be nan (e.g. the first step)."""
    first_step = transition.steps[0]
    if math.isnan(first_step.gamma):
        return False
    elif torch.isnan(first_step.state).any():
        return False
    elif math.isnan(first_step.dp):
        return False

    for step in transition.steps[1:]:
        if has_nan(step):
            return False

    return True

