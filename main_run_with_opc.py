import logging
import time
from random import random

import pandas as pd
from pydantic import Field

from corerl.configs.config import config
from corerl.configs.loader import load_config
from corerl.data_pipeline.pipeline import PipelineConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.deployment_async_env import DepAsyncEnvConfig, DeploymentAsyncEnv


@config(allow_extra=True)
class Config:
    env: DepAsyncEnvConfig = Field(default_factory=DepAsyncEnvConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


def dumb_policy(action_tags: list[TagConfig]):
    raw_actions = {}
    for action_tag in action_tags:
        assert action_tag.operating_range is not None
        lo, hi = action_tag.operating_range
        lo = lo if lo is not None else 0
        hi = hi if hi is not None else 1
        raw_actions[action_tag.name] = [(hi - lo) * random() + lo]
    return pd.DataFrame(raw_actions)


@load_config(Config, base="config/")
def main(cfg: Config):
    dep_env = DeploymentAsyncEnv(cfg.env, cfg.pipeline.tags)

    action_tags = [tag_config for tag_config in cfg.pipeline.tags if tag_config.action_constructor is not None]
    for _ in range(1000):
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
