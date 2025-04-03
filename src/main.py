import gymnasium as gym
import jax
from tqdm import tqdm

from agent.gac import GreedyAC, GreedyACConfig
from interaction.env_wrapper import EnvWrapper
from interaction.logging import log_to_file, setup_logging


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

    agent = GreedyAC(GreedyACConfig(),
                     seed=0,
                     state_dim=wrapper_env.get_state_dim(),
                     action_dim=1)
    rng = jax.random.PRNGKey(0)

    episodes = 2
    for episode in range(episodes):
        state, _ = wrapper_env.reset()
        episode_reward = 0.0
        steps = 0
        pbar = tqdm()
        while True:
            rng, step_key = jax.random.split(rng)
            action = agent.get_actions(state)

            next_state, reward, terminated, truncated, info, transitions = wrapper_env.step(action)
            for t in transitions:
                agent.update_buffer(t)

            agent.update()
            state_dict = {f"state_{i}": float(x) for i, x in enumerate(next_state)}
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
            pbar.update(1)

            if terminated or truncated:
                break

            state = next_state

    env.close()

if __name__ == "__main__":
    main()
