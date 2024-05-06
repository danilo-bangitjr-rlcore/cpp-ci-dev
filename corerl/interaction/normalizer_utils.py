from omegaconf import DictConfig
from abc import ABC, abstractmethod

import numpy as np
import gymnasium


class BaseNormalizer(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        raise NotImplementedError


class Identity(BaseNormalizer):
    def __init__(self):
        return

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        return x

    def denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        return x


class Scale(BaseNormalizer):
    def __init__(self, scale: float | np.ndarray, bias: float | np.ndarray):
        self.scale = scale
        self.bias = bias

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        return (x - self.bias) / self.scale

    def denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        return x * self.scale + self.bias


class Clip(BaseNormalizer):
    def __init__(self, min_: float | np.ndarray, max_: float | np.ndarray):
        self.min = min_
        self.max = max_

    def __call__(self, x: float | np.ndarray) -> np.float64 | np.ndarray:
        return np.clip(x, self.min, self.max)

    def denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        return x

class OneHot(BaseNormalizer):
    def __init__(self, total_count: int, start_from: int):
        self.total_count = total_count
        self.start = start_from

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2 and x.shape[1] == 1 # shape = batch_size * 1
        oneh = np.zeros((x.shape[0], self.total_count))
        oneh[np.arange(x.shape[0]), (x - self.start).astype(int).squeeze()] = 1
        return oneh

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        idx = np.where(x==1)[1]
        idx = np.expand_dims(idx, axis=1)
        return idx


def init_action_normalizer(cfg: DictConfig, env: gymnasium.Env) -> BaseNormalizer:
    if cfg.discrete_control:
        return Identity()
    else:  # continuous control
        name = cfg.name
        if name == "identity":
            return Identity()
        elif name == "scale":
            if cfg.action_low:  # use high and low specified in the config
                assert cfg.action_high, "Please specify cfg.action_normalizer.action_high in config file"
                action_low = np.array(cfg.action_low)
                action_high = np.array(cfg.action_high)
            else:  # use information from the environment
                action_low = env.action_space.low
                action_high = env.action_space.high
            scale = action_high - action_low
            bias = action_low
            return Scale(scale, bias)
        elif name == "clip":
            return Clip(cfg.clip_min, cfg.clip_max)
        else:
            raise NotImplementedError


def init_reward_normalizer(cfg: DictConfig) -> BaseNormalizer:
    name = cfg.name
    if name == "identity":
        return Identity()
    elif name == "scale":
        assert cfg.reward_high, "Please specify cfg.reward_normalizer.reward_high in config file"
        reward_high = float(cfg.reward_high)
        reward_low = float(cfg.reward_low)
        scale = reward_high - reward_low
        bias = float(cfg.bias)
        return Scale(scale, bias)
    elif name == "clip":
        return Clip(cfg.clip_min, cfg.clip_max)
    else:
        raise NotImplementedError
