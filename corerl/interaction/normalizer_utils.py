import warnings

from omegaconf import DictConfig
from abc import ABC, abstractmethod

import numpy as np
import gymnasium

import corerl.state_constructor.components as sc


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
        assert len(x.shape) == 2 and x.shape[1] == 1  # shape = batch_size * 1
        oneh = np.zeros((x.shape[0], self.total_count))
        oneh[np.arange(x.shape[0]), (x - self.start).astype(int).squeeze()] = 1
        return oneh

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        idx = np.where(x == 1)[1]
        idx = np.expand_dims(idx, axis=1)
        return idx


class MaxMin(BaseNormalizer):
    def __init__(self, env):
        self.min = env.observation_space.low
        self.max = env.observation_space.high
        self.scale = self.max - self.min
        self.bias = self.min

    def __call__(self, x: float | np.ndarray) -> np.float64 | np.ndarray:
        return (x - self.bias) / self.scale

    def denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        return x * self.scale + self.bias


class AvgNanNorm(BaseNormalizer):
    def __init__(self, env):
        self.min = env.observation_space.low
        self.max = env.observation_space.high
        self.scale = self.max - self.min
        self.bias = self.min

    def __call__(self, x: float | np.ndarray) -> np.float64 | np.ndarray:
        x = self.handle_nan(x)
        # x = self.average(x)
        return self.normalize(x)

    def normalize(self, x: float | np.ndarray) -> float | np.ndarray:
        return (x - self.bias) / self.scale

    def denormalize(self, x: float | np.ndarray) -> float | np.ndarray:
        return x * self.scale + self.bias

    def handle_nan(self, x: np.ndarray) -> np.ndarray:
        # Adapted from https://stackoverflow.com/a/62039015
        fill = np.nanmean(x, axis=0)  # try to fill with the mean in each column
        # If an entire column is nan, fill with zeros
        fill_mask = np.isnan(fill)
        fill[fill_mask] = np.zeros_like(fill)[fill_mask]

        mask = np.isnan(x[0])
        x[0][mask] = fill[mask]
        for i in range(1, len(x)):
            mask = np.isnan(x[i])
            x[i][mask] = x[i - 1][mask]
        return x

    def average(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 2:
            return np.mean(x, axis=0)
        elif len(x.shape) == 1:
            return x
        else:
            raise ValueError("Invalid shape for observation")


def _get_policy_bounds(agent):
    return agent.actor.distribution_bounds()


def init_action_normalizer(cfg: DictConfig, env: gymnasium.Env) -> BaseNormalizer:
    if cfg.discrete_control:
        return Identity()
    else:  # continuous control
        name = cfg.name
        if name == "identity":
            action_min = env.action_space.low
            action_max = env.action_space.high

            warnings.warn(
                "\033[1;33m" +
                f"actions are bounded between [{action_min}, {action_max}] " +
                f"but the policy has support only over [0, 1]. Are you sure this is what you wanted to do?" +
                "\033[0m")

            return Identity()
        elif name == "scale":
            if cfg.use_cfg_values:  # use high and low specified in the config
                action_min = np.array(cfg.action_low)
                action_max = np.array(cfg.action_high)
            else:  # use information from the environment
                action_min = env.action_space.low
                action_max = env.action_space.high

            scale = (action_max - action_min)
            bias = action_min

            print(f"Using scale = {scale} and bias = {bias}")

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
        reward_high = float(cfg.reward_high)
        reward_low = float(cfg.reward_low)
        scale = reward_high - reward_low
        bias = float(cfg.reward_bias)
        return Scale(scale, bias)
    elif name == "clip":
        return Clip(cfg.clip_min, cfg.clip_max)
    else:
        raise NotImplementedError


def init_obs_normalizer(cfg: DictConfig, env) -> BaseNormalizer:
    name = cfg.name
    if name == "identity":
        return Identity()
    elif name == "maxmin":
        return MaxMin(env)
    elif name == "avg_nan_norm":
        return AvgNanNorm(env)
