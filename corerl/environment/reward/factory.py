from typing import Any

from corerl.environment.reward.base import BaseReward
from corerl.environment.reward.reseau import ReseauReward
from corerl.environment.reward.saturation import SaturationReward
from corerl.environment.reward.scrubber import ScrubberReward


def init_reward_function(cfg: Any) -> BaseReward:
    name = cfg.name
    if name == "reseau":
        return ReseauReward(cfg)
    elif name == 'saturation':
        return SaturationReward(cfg)
    elif name == 'scrubber':
        return ScrubberReward(cfg)
    else:
        raise NotImplementedError
