import gymnasium as gym
import numpy as np

from src.interaction.env_wrapper import EnvWrapper


def test_env_wrapper_mountain_car():
    env = gym.make('MountainCarContinuous-v0')
    observation_space_info = {
        'low': np.array([-1.2, -0.07]),
        'high': np.array([0.6, 0.07])
    }

    wrapper = EnvWrapper(
        env=env,
        observation_space_info=observation_space_info,
        min_n_step=1,
        max_n_step=3,
        gamma=0.99
    )

    state, info = wrapper.reset()
    assert isinstance(state, dict)
    assert '0' in state
    assert '1' in state
    print(state)


    state_array = wrapper.to_array(state)
    assert isinstance(state_array, np.ndarray)
    print(state_array)
    assert state_array.shape == (2,)

    # check normalization
    state_array = wrapper.to_array(state)
    assert np.all(state_array >= 0.0) and np.all(state_array <= 1.0), "States should be normalized between 0 and 1"

    # check n-step transitions
    transitions_list = []
    for _ in range(5):
        action = np.array([1.0])
        state, reward, terminated, truncated, info, transitions = wrapper.step(action)
        print(transitions)
        transitions_list.extend(transitions)
        state_array = wrapper.to_array(state)
        assert np.all(state_array >= 0.0) and np.all(state_array <= 1.0), "States should be normalized between 0 and 1"

        if terminated or truncated:
            break

    n_steps = set(t.n_steps for t in transitions_list)
    assert len(n_steps) > 1
    assert max(n_steps) <= 3
    assert min(n_steps) >= 1



