import gymnasium as gym
import jax.numpy as jnp

from src.interaction.env_wrapper import EnvWrapper


def test_env_wrapper_mountain_car():
    env = gym.make('MountainCarContinuous-v0')
    observation_space_info = {
        'low': jnp.array([-1.2, -0.07]),
        'high': jnp.array([0.6, 0.07])
    }

    wrapper = EnvWrapper(
        env=env,
        observation_space_info=observation_space_info,
        min_n_step=1,
        max_n_step=1,
        gamma=0.99
    )

    state, info = wrapper.reset()
    assert isinstance(state, jnp.ndarray)
    assert state.shape == (2,)

    assert jnp.all(state >= 0.0) and jnp.all(state <= 1.0), "States should be normalized between 0 and 1"

    transitions_list = []
    for _ in range(5):
        action = jnp.array([1.0])
        state, _, terminated, truncated, _, transitions = wrapper.step(action)
        transitions_list.extend(transitions)
        assert jnp.all(state >= 0.0) and jnp.all(state <= 1.0), "States should be normalized between 0 and 1"

        if terminated or truncated:
            break

    n_steps = set(t.n_steps for t in transitions_list)
    assert max(n_steps) == 1
    assert min(n_steps) == 1


