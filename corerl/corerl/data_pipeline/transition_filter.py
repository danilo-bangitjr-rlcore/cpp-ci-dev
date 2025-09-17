import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, assert_never

import jax
import jax.numpy as jnp
from lib_config.config import config, list_, post_processor

from corerl.data_pipeline.datatypes import PipelineFrame, Transition
from corerl.state import AppState

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
    def _validate_compatible_filters(self, cfg: "MainConfig"):
        if "only_dp" in self.filters:
            assert (
                'only_post_dp' not in self.filters
            ), "'only_dp' and 'only_post_dp' are inconsistent transition filters."

            assert (
                'only_pre_dp_or_ac' not in self.filters
            ), "'only_dp' and 'only_pre_dp_or_ac' are inconsistent transition filters."


class TransitionFilter:
    def __init__(self, app_state: AppState, cfg: TransitionFilterConfig):
        self.filter_names = cfg.filters
        self._app_state = app_state

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        if pf.transitions is None:
            return pf

        transitions_before = len(pf.transitions)
        for filter_name in self.filter_names:
            pf.transitions = call_filter(pf.transitions, filter_name)

            transitions_after = len(pf.transitions)
            self._app_state.metrics.write(
                self._app_state.agent_step,
                f'transitions_filtered_by_{filter_name}',
                transitions_before - transitions_after,
            )
            transitions_before = transitions_after

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
    for value in vars(obj).values():
        if isinstance(value, jax.Array):
            if jnp.isnan(value).any():
                return True
        elif isinstance(value, float) and math.isnan(value):
            return True
    return False


def no_nan(transition: Transition):
    return not any(has_nan(step) for step in transition.steps)
