import gymnasium as gym
import jax

from src.agent.random import RandomAgent
from src.interaction.env_wrapper import EnvWrapper
from src.interaction.logging import log_to_file, setup_logging


def main():
    log_path = setup_logging()

    env = gym.make("MountainCarContinuous-v0")

    wrapper_env = EnvWrapper(
        env=env,
        observation_space_info={
            'low': env.observation_space.low,
            'high': env.observation_space.high
        },
        min_n_step=1,
        max_n_step=1,
        gamma=0.99
    )

    agent = RandomAgent(seed=0, action_dim=1)
    rng = jax.random.PRNGKey(0)

    episodes = 2
    for episode in range(episodes):
        state, _ = wrapper_env.reset()
        episode_reward = 0.0
        steps = 0

        while True:
            rng, step_key = jax.random.split(rng)
            state_array = wrapper_env.to_array(state).reshape(1, -1)
            action = agent.get_actions(step_key, state_array)[0]
            agent.update()

            next_state, reward, terminated, truncated, info, transitions = wrapper_env.step(action)

            state_dict = {f"state_{i}": float(x) for i, x in enumerate(wrapper_env.to_array(next_state))}
            action_dict = {f"action_{i}": float(x) for i, x in enumerate(action)}

            log_to_file(
                log_path,
                measurement="gym_environment",
                tags={
                    "env_id": env.unwrapped.spec.id,
                    "episode": episode,
                    "step": steps
                },
                fields={
                    **state_dict,
                    **action_dict,
                    "reward": float(reward),
                    "episode_reward": float(episode_reward),
                }
            )

            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

            state = next_state

    env.close()

if __name__ == "__main__":
    main()
