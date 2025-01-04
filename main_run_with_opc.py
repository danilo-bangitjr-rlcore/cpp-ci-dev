import logging
import time
from dataclasses import field
from random import random

import numpy as np

from corerl.configs.config import config
from corerl.configs.loader import load_config
from corerl.data_pipeline.pipeline import PipelineConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.deployment_async_env import DepAsyncEnvConfig, DeploymentAsyncEnv


@config(allow_extra=True)
class Config:
    env: DepAsyncEnvConfig = field(default_factory=DepAsyncEnvConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def dumb_policy(action_tags: list[TagConfig]):
    raw_actions = []
    for action_tag in action_tags:
        lo, hi = action_tag.bounds
        lo = lo if lo is not None else 0
        hi = hi if hi is not None else 1
        raw_actions.append((hi - lo) * random() + lo)
    return np.ndarray(raw_actions)


@load_config(Config, base="config/")
def main(cfg: Config):
    dep_env = DeploymentAsyncEnv(cfg.env, cfg.pipeline.tags)

    for _ in range(1000):
        action_tags = [tag_config for tag_config in cfg.pipeline.tags if tag_config.is_action]
        action = dumb_policy(action_tags)
        dep_env.emit_action(action)
        time.sleep(1)
        print(dep_env.get_latest_obs())

        # if terminated or truncated:  # TODO: Still not sure how we handle this
        #     observation, info = env.reset()

    dep_env.close()


if __name__ == "__main__":
    _logger = logging.getLogger(__name__)
    main()
