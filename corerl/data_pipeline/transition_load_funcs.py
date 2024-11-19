"""
A library of functions to transform lists of observation transitions into transitions or trajectories
"""
import numpy as np

from tqdm import tqdm

from corerl.data.data import ObsTransition, Transition
from corerl.state_constructor.base import BaseStateConstructor
from corerl.data.transition_creator import BaseTransitionCreator
from corerl.data.obs_normalizer import ObsTransitionNormalizer


def make_transitions(
    obs_transitions: list[ObsTransition],
    sc: BaseStateConstructor,
    tc: BaseTransitionCreator,
    obs_normalizer: ObsTransitionNormalizer,
    warmup=0,
) -> list[Transition]:
    obs_transitions = _normalize_obs_transitions(obs_transitions, obs_normalizer)
    curr_chunk_obs_transitions = []
    transitions = []

    last_idx = len(obs_transitions) - 1
    regular_rl_warmup = warmup // tc.steps_per_decision
    transition_kind = tc.transition_kind
    for idx, obs_transition in enumerate(tqdm(obs_transitions)):
        curr_chunk_obs_transitions.append(obs_transition)
        done = (idx == last_idx)
        if obs_transition.gap or done:
            new_transitions = make_transitions_for_chunk(curr_chunk_obs_transitions, sc, tc)
            if transition_kind == 'anytime':
                transitions += new_transitions[warmup:]
            elif transition_kind == 'regular_rl':
                transitions += new_transitions[regular_rl_warmup:]
            else:
                raise AssertionError('Unknown transition kind: {}'.format(transition_kind))

            curr_chunk_obs_transitions = []

    return transitions


def _normalize_obs_transitions(
    obs_transitions: list[ObsTransition],
    obs_normalizer: ObsTransitionNormalizer,
) -> list[ObsTransition]:
    return [obs_normalizer.normalize(ot) for ot in obs_transitions]


def _get_action_windows(obs_transitions: list[ObsTransition]) -> list[list[ObsTransition]]:
    """
    Breaks obs_transitions into a list of lists, where each element of the list is a maximal list of obs transitions
    with the same action (i.e. an action window)
    """
    curr_action = obs_transitions[0].action
    action_windows = []
    curr_action_obs_transitions = []

    for obs_transition in obs_transitions:
        if obs_transition.action == curr_action:
            curr_action_obs_transitions.append(obs_transition)

        else:
            action_windows.append(curr_action_obs_transitions)
            curr_action_obs_transitions = [obs_transition]
            curr_action = obs_transition.action

    action_windows.append(curr_action_obs_transitions)  # append any left-overs as a new action window

    # check that everything is correct
    assert sum([len(aw) for aw in action_windows]) == len(obs_transitions)  # is a partition
    prev_action = None
    for action_window in action_windows:
        # check all elements of the action window has the same action
        action = action_window[0].action
        for obs_t in action_window:
            assert action == obs_t.action
        assert action != prev_action  # neighbouring action windows have different actions
        prev_action = action

    return action_windows


def _get_right_aligned_action_window_dps(
    action_window: list[ObsTransition],
    steps_per_decision: int,
) -> tuple[list[bool], list[int]]:
    """
    Iterates through an action window and determines whether reach obs transition is a decision point (dp)
    and what the number of steps until the next decision are for that obs transition.

    Right aligned means that any remainder of the action window that is not divisible by steps_per_decision are
    treated as a decision window at the START of the action window.
    """
    num_action_steps = len(action_window)
    remainder_steps = (num_action_steps % steps_per_decision)
    steps_until_decision = remainder_steps

    aw_dps = list()
    aw_steps_until_decisions = list()
    dp = True  # true initially, by convention

    for _ in action_window:
        if steps_until_decision == 0:
            steps_until_decision = steps_per_decision
            dp = True

        aw_dps.append(dp)
        aw_steps_until_decisions.append(steps_until_decision)
        steps_until_decision -= 1
        dp = False

    return aw_dps, aw_steps_until_decisions


def _get_left_aligned_action_window_dps(
    action_window: list[ObsTransition],
    steps_per_decision: int,
) -> tuple[list[bool], list[int]]:
    """
    Iterates through an action window and determines whether reach obs transition is a decision point (dp)
    and what the number of steps until the next decision are for that obs transition.

    left aligned means that any remainder of the action window that is not divisible by steps_per_decision are
    treated as a decision window at the END of the action window.
    """

    num_action_steps = len(action_window)
    remainder_steps = (num_action_steps % steps_per_decision)
    steps_until_decision = steps_per_decision if num_action_steps >= steps_per_decision else remainder_steps
    aw_dps = list()
    aw_steps_until_decisions = list()
    dp = True  # true initially, by convention
    for i in range(num_action_steps):
        if steps_until_decision == 0:
            dp = True
            within_last_decision_point = i >= num_action_steps - remainder_steps
            if within_last_decision_point:
                steps_until_done = num_action_steps - i
                steps_until_decision = steps_until_done
            else:
                steps_until_decision = steps_per_decision

        aw_dps.append(dp)
        aw_steps_until_decisions.append(steps_until_decision)
        steps_until_decision -= 1
        dp = False
    return aw_dps, aw_steps_until_decisions


def _get_last_dp(steps_until_decisions, steps_per_decision):
    """
    Returns the dp for the final observation
    """
    last_steps_until_decision = steps_until_decisions[-1] - 1
    if last_steps_until_decision == 0:
        last_steps_until_decision = steps_per_decision
    elif last_steps_until_decision == -1:
        last_steps_until_decision = steps_per_decision - 1

    last_dp = last_steps_until_decision == steps_per_decision
    return last_dp, last_steps_until_decision


def _get_dps_and_steps_until_decision(
    obs_transitions: list[ObsTransition],
    steps_per_decision: int,
    right_align=True,
) -> tuple[list[bool], list[int]]:
    """
    Returns two lists:
        1. Whether each observation is a decision point.
        2. The number of steps until the next decision.

    The ith element in each of these lists refers to the FIRST observation in the ith element of obs_transitions.

    The only exception is the last element of dps/steps_until_decision, which refers to the NEXT observation of the last
    element of obs_transitions.

    Thus, these dps and steps_until_decisions are of length len(obs_transitions) + 1
    """

    action_windows = _get_action_windows(obs_transitions)

    dps, steps_until_decisions = [], []
    for action_window in action_windows:
        if right_align:
            aw_dps, aw_steps_until_decisions = _get_right_aligned_action_window_dps(action_window, steps_per_decision)
        else:
            aw_dps, aw_steps_until_decisions = _get_left_aligned_action_window_dps(action_window, steps_per_decision)
        dps += aw_dps
        steps_until_decisions += aw_steps_until_decisions

    last_dp, last_steps_until_decision = _get_last_dp(steps_until_decisions, steps_per_decision)
    dps.append(last_dp)
    steps_until_decisions.append(last_steps_until_decision)

    assert len(steps_until_decisions) == len(dps) == len(obs_transitions) + 1

    return dps, steps_until_decisions


def make_transitions_for_chunk(
    obs_transitions: list[ObsTransition],
    sc: BaseStateConstructor,
    tc: BaseTransitionCreator,
) -> list[Transition]:
    """
    Given a list of obs transitions, return a list of transitions for this list.
    """
    steps_per_decision = tc.steps_per_decision
    for i in range(len(obs_transitions) - 1):
        assert np.allclose(obs_transitions[i].next_obs, obs_transitions[i + 1].obs)
        assert not obs_transitions[i].gap

    sc.reset()
    dps, steps_until_decisions = _get_dps_and_steps_until_decision(obs_transitions, steps_per_decision)

    initial_obs_transition = obs_transitions[0]
    initial_steps_until_decision = steps_until_decisions[0]
    initial_dp = dps[0]
    initial_state = sc(initial_obs_transition.obs,
        initial_obs_transition.action,  # assume that previous action was the same as the next
        initial_state=True,
        decision_point=initial_dp,
        steps_until_decision=initial_steps_until_decision)

    tc.reset(initial_state, initial_dp, initial_steps_until_decision)

    transitions = []
    for obs_idx, obs_transition in enumerate(obs_transitions):
        next_dp = dps[obs_idx + 1]
        next_steps_until_decision = steps_until_decisions[obs_idx + 1]

        next_state = sc(obs_transition.next_obs, obs_transition.action,
            initial_state=False, decision_point=next_dp, steps_until_decision=next_steps_until_decision)

        transitions += tc.feed(obs_transition, next_state, next_dp=next_dp,
            next_steps_until_decision=next_steps_until_decision)

    return transitions
