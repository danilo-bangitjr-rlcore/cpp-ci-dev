from omegaconf import DictConfig

from corerl.environment.reward.base import BaseReward
from corerl.environment.reward.reseau import ReseauReward

def init_reward_function(cfg: DictConfig) -> BaseReward:
    name = cfg.name
    if name == "reseau":
        return ReseauReward(cfg)
    else:
        raise NotImplementedError