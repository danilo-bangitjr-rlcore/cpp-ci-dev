import pytest
import numpy as np
from omegaconf import DictConfig

from corerl.data.transition_creator import AnytimeTransitionCreator
from corerl.data.data import ObsTransition
from test.unit.state_constructor.state_constructor import make_anytime_multi_trace
from corerl.state_constructor.base import BaseStateConstructor


def _make_anytime_transition_creator(
    sc: BaseStateConstructor,
    steps_per_decision: int,
    n_step: int,
) -> AnytimeTransitionCreator:
    cfg_d = {
        'steps_per_decision': steps_per_decision,
        'n_step': n_step,
        'gamma': 0.9,
        'transition_kind': 'anytime'
    }
    cfg = DictConfig(cfg_d)
    return AnytimeTransitionCreator(cfg, sc)


def _make_simple_obs_sequence(num_observations: int) -> list[np.ndarray]:
    obs_sequence = [np.array([1])]
    for i in range(num_observations - 1):
        obs_sequence.append(np.array([0]))
    return obs_sequence


def _make_simple_obs_transition_sequence(num_observations: int):
    observations = _make_simple_obs_sequence(num_observations)
    action = np.array([1])
    reward = 1

    obs_transitions = []
    for i in range(len(observations) - 1):
        new_obs_transition = ObsTransition(
            obs=observations[i],
            action=action,
            reward=reward,
            next_obs=observations[i + 1],
            terminated=False,
            truncate=False,
            gap=False
        )
        obs_transitions.append(new_obs_transition)

    return obs_transitions


def _get_first_state(sc: BaseStateConstructor, obs_transitions: list[ObsTransition]):
    sc.reset()
    initial_obs = obs_transitions[0].obs
    initial_action = obs_transitions[0].obs
    dummy_action = np.zeros_like(initial_action)
    state = sc(initial_obs, dummy_action, initial_state=True, decision_point=True)
    return sc, state


def test_anytime_transition_creator_init():
    make_anytime_multi_trace(warmup=0, steps_per_decision=5)


def _check_dps(i, steps_per_decision, transition, n_step):
    steps_until_decision = steps_per_decision - (i % steps_per_decision)
    # check if state_dp set properly
    if i % steps_per_decision == 0:
        assert transition.state_dp

    # check if boot_state_dp set properly
    if n_step == 0:
        assert transition.boot_state_dp
    else:
        if n_step >= steps_until_decision:
            assert transition.boot_state_dp

    # check if next_state_dp set properly
    if steps_until_decision == 1:
        assert transition.next_state_dp


@pytest.mark.parametrize("num_observations, steps_per_decision, n_step",
    [
        (10, 1, 0),
        (21, 2, 0),
        (1001, 5, 0),
        (101, 10, 0),
        (100, 3, 1),
        (10, 5, 2),

    ])
def test_anytime_transition_creator_feed(
        num_observations,
        steps_per_decision,
        n_step
):
    """
    Given a state constructor and a sequence of observations,
    the anytime transition creator creates a sequence of transitions
    and correctly marks which of those transitions are decision points.
    """
    sc = make_anytime_multi_trace(warmup=0, steps_per_decision=steps_per_decision)
    tc = _make_anytime_transition_creator(sc, steps_per_decision, n_step)

    obs_transitions = _make_simple_obs_transition_sequence(num_observations)
    sc, state = _get_first_state(sc, obs_transitions)

    steps_until_decision = steps_per_decision

    tc.reset(state, dp=True, steps_until_decision=steps_until_decision)
    transitions = []

    for i, obs_transition in enumerate(obs_transitions):
        steps_until_decision -= 1

        if steps_until_decision == 0:
            decision_point = True
            steps_until_decision = steps_per_decision
        else:
            decision_point = False

        next_state = sc(obs_transition.next_obs, obs_transition.action, decision_point=decision_point)
        new_transitions = tc.feed(
            obs_transition=obs_transition,
            next_state=next_state,
            next_dp=decision_point,
            next_steps_until_decision=steps_until_decision)
        transitions += new_transitions

        if decision_point:
            assert len(new_transitions)
        else:
            assert not len(new_transitions)

    for i, transition in enumerate(transitions):
        last_transition = i == len(transitions) - 1
        transition = transitions[i]
        _check_dps(i, steps_per_decision, transition, n_step)

        if not last_transition:
            next_transition = transitions[i + 1]
            assert np.allclose(transition.next_state, next_transition.state)
