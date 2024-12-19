from dataclasses import field
import gymnasium as gym
import logging
import time

from corerl.configs.config import config
from corerl.configs.loader import load_config
from corerl.environment.async_env.deployment_async_env import DepAsyncEnvConfig, DeploymentAsyncEnv
from corerl.environment.config import EnvironmentConfig


@config(allow_extra=True)
class MainConfig:
    env: DepAsyncEnvConfig = field(default_factory=DepAsyncEnvConfig)


# We need this for our dumb agent, but right now it is a little cheat
def get_action_and_obs_spaces(cfg: EnvironmentConfig):
    env = gym.make(cfg.name)
    action_space = env.action_space
    observation_space = env.observation_space
    env.close()
    return action_space, observation_space


def dumb_policy(action_space: gym.Space):
    return action_space.sample()


@load_config(MainConfig, base='config/')
def main(cfg: MainConfig):
    action_space, obs_space = get_action_and_obs_spaces(cfg.env)
    dep_env = DeploymentAsyncEnv(cfg.env)

    for _ in range(1000):
        action = dumb_policy(action_space)
        dep_env.emit_action(action)
        time.sleep(1)
        print(dep_env.get_latest_obs())

        # if terminated or truncated:  # TODO: Still not sure how we handle this
        #     observation, info = env.reset()

    dep_env.close()

if __name__ == "__main__":
    _logger = logging.getLogger(__name__)
    main()
