import gymnasium as gym
import logging
import time

import hydra
from omegaconf import DictConfig

from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv

# Imports from main
# from corerl.environment.factory import init_environment
from corerl.config import MainConfig  # noqa: F401
from corerl.utils.device import device  # noqa: F401
from corerl.agent.factory import init_agent  # noqa: F401
from corerl.utils.plotting import make_online_plots, make_offline_plots  # noqa: F401
from corerl.environment.reward.factory import init_reward_function  # noqa: F401
from corerl.agent.base import BaseAgent  # noqa: F401
from corerl.utils.plotting import make_actor_critic_plots, make_reseau_gvf_critic_plot  # noqa: F401
import corerl.utils.dict as dict_u  # noqa: F401
import corerl.utils.nullable as nullable  # noqa: F401



# We need this for our dumb agent, but right now it is a little cheat
def get_action_and_obs_spaces(cfg):
    env = gym.make(*cfg.environment.args, **cfg.environment.kwargs)
    action_space = env.action_space
    observation_space = env.observation_space
    env.close()
    return action_space, observation_space


def dumb_policy(action_space: gym.Space):
    return action_space.sample()


@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: DictConfig):
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
