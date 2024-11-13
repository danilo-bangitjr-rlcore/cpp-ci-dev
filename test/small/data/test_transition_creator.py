import pytest
import numpy as np

from corerl.data.data import ObsTransition, Transition
from corerl.data.transition_creator import AnytimeTransitionCreator, RegularRLTransitionCreator, AnytimeTCConfig, RegularRLTCConfig # noqa: E501
from corerl.state_constructor.examples import IdentityConfig
from corerl.state_constructor.factory import init_state_constructor
from test.small.state_constructor.state_constructor import make_anytime_multi_trace
from test.infrastructure.dummy_data.transitions import make_simple_obs_transition_sequence



def test_regular_rl_transition_creator():
    sc = init_state_constructor(IdentityConfig())
    cfg = RegularRLTCConfig(
        steps_per_decision=5,
        n_step=1,
        gamma=0.1,
    )
    tc = RegularRLTransitionCreator(
        cfg=cfg,
        state_constuctor=sc,
    )

    # transform first observation into a state
    state = sc(
        obs=np.array([100]),
        action=np.array([0]),
        initial_state=True,
    )
    tc.reset(state, dp=True, steps_until_decision=5)

    # the next 4 obs transitions are interior to the decision window
    # and so will not produce a transition
    for i in range(4):
        next_transition = ObsTransition(
            obs=np.array([100 + i]),
            action=np.array([0]),
            reward=1.0,
            next_obs=np.array([101 + i]),
            terminated=False,
            truncate=False,
            gap=False,
        )
        next_state = sc(
            obs=next_transition.next_obs,
            action=next_transition.action
        )
        transitions = tc.feed(
            obs_transition=next_transition,
            next_state=next_state,
            next_dp=False,
            next_steps_until_decision=4 - i,
        )
        assert len(transitions) == 0

    # the next obs transition is a decision point
    # so a new transition should be created
    next_transition = ObsTransition(
        obs=np.array([104]),
        action=np.array([0]),
        reward=1.0,
        next_obs=np.array([105]),
        terminated=False,
        truncate=False,
        gap=False,
    )
    next_state = sc(
        obs=next_transition.next_obs,
        action=next_transition.action
    )
    transitions = tc.feed(
        obs_transition=next_transition,
        next_state=next_state,
        next_dp=True,
        next_steps_until_decision=0,
    )

    assert len(transitions) == 1
    assert transitions[0] == Transition(
        obs=np.array([100]),
        state=np.array([1, 0, 100]),
        action=np.array([0]),
        next_obs=np.array([101]),
        next_state=np.array([0, 0, 101]),
        reward=1.0,
        n_step_reward=1.1111,
        n_step_cumulants=None,
        boot_obs=np.array([105]),
        boot_state=np.array([0, 0, 105]),
        terminated=False,
        truncate=False,
        state_dp=True,
        next_state_dp=False,
        boot_state_dp=True,
        gamma_exponent=5,
        gap=False,
        steps_until_decision=5,
        next_steps_until_decision=4,
        boot_steps_until_decision=5,
    )


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
    cfg = AnytimeTCConfig(
        gamma=0.9,
        steps_per_decision=steps_per_decision,
        n_step=n_step,
    )
    tc = AnytimeTransitionCreator(
        cfg=cfg,
        state_constuctor=sc,
    )

    obs_transitions = make_simple_obs_transition_sequence(num_observations)

    initial_obs = obs_transitions[0].obs
    initial_action = obs_transitions[0].action
    dummy_action = np.zeros_like(initial_action)
    state = sc(initial_obs, dummy_action, initial_state=True, decision_point=True)

    steps_until_decision = steps_per_decision

    tc.reset(state, dp=True, steps_until_decision=steps_until_decision)
    transitions = []

    for _, obs_transition in enumerate(obs_transitions):
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
            assert len(new_transitions) > 0
        else:
            assert len(new_transitions) == 0

    for i, transition in enumerate(transitions):
        last_transition = i == len(transitions) - 1
        transition = transitions[i]
        _check_dps(i, steps_per_decision, transition, n_step)

        if not last_transition:
            next_transition = transitions[i + 1]
            assert np.allclose(transition.next_state, next_transition.state)


def _check_dps(i: int, steps_per_decision: int, transition: Transition, n_step: int):
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
