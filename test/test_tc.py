from src.interaction.transition_creator import Step, TransitionCreator


def make_test_step(state_num: int, gamma: float = 0.9) -> Step:
    return Step(
        state=state_num,
        action=0,
        reward=1,
        next_state=state_num + 1,
        done=False,
        gamma=gamma
    )


def test_get_n_step_reward():
    tc = TransitionCreator(min_n_step=2, max_n_step=2, gamma=0.9)

    step_0 = make_test_step(0)
    step_1 = make_test_step(1)
    step_2 = make_test_step(2)

    transitions = tc(step_0.state, step_0.action, step_0.reward, step_0.next_state, step_0.done)
    assert len(transitions) == 0

    transitions = tc(step_1.state, step_1.action, step_1.reward, step_1.next_state, step_1.done)
    assert len(transitions) == 0

    transitions = tc(step_2.state, step_2.action, step_2.reward, step_2.next_state, step_2.done)
    assert len(transitions) == 1

    transition = transitions[0]
    assert transition.n_step_reward == 1.9  # reward_1 + 0.9 * reward_2
    assert transition.n_step_gamma == 0.81  # 0.9 * 0.9


def test_basic_transitions():
    tc = TransitionCreator(min_n_step=1, max_n_step=2, gamma=0.9)

    transitions = []
    for i in range(3):
        step = make_test_step(i)
        new_transitions = tc(step.state, step.action, step.reward, step.next_state, step.done)
        transitions.extend(new_transitions)

    assert len(transitions) == 3  # 2 one-step transitions + 1 two-step transition

    # Test 1-step transitions
    assert transitions[0].steps[0].state == 0
    assert transitions[0].n_step_reward == 1.0
    assert transitions[0].n_step_gamma == 0.9

    assert transitions[1].steps[0].state == 1
    assert transitions[1].n_step_reward == 1.0
    assert transitions[1].n_step_gamma == 0.9

    # Test 2-step transition
    assert transitions[2].steps[0].state == 0
    assert transitions[2].n_step_reward == 1.9  # 1 + 0.9*1
    assert transitions[2].n_step_gamma == 0.81


def test_episode_reset():
    tc = TransitionCreator(min_n_step=1, max_n_step=2, gamma=0.9)
    step_0 = make_test_step(0)
    step_1 = make_test_step(1)
    step_1 = Step(**{**step_1.__dict__, 'done': True})

    transitions = tc(step_0.state, step_0.action, step_0.reward, step_0.next_state, step_0.done)
    assert len(transitions) == 0

    transitions = tc(step_1.state, step_1.action, step_1.reward, step_1.next_state, step_1.done)
    assert len(transitions) == 2  # one 1-step and one 2-step transition

    # Start new episode
    step_2 = make_test_step(2)
    transitions = tc(step_2.state, step_2.action, step_2.reward, step_2.next_state, step_2.done)
    assert len(transitions) == 0  # buffer was reset due to episode end

def test_min_n_step_only():
    tc = TransitionCreator(min_n_step=2, max_n_step=2, gamma=0.9)

    transitions = []
    for i in range(5):
        step = make_test_step(i)
        new_transitions = tc(step.state, step.action, step.reward, step.next_state, step.done)
        transitions.extend(new_transitions)

    assert len(transitions) == 3

    for i, transition in enumerate(transitions):
        assert len(transition.steps) == 3
        assert transition.steps[0].state == i
        assert transition.n_step_reward == 1.9
        assert transition.n_step_gamma == 0.81
