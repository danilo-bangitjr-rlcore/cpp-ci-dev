import logging
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PerturbationConfig:
    enabled: bool = False
    frequency: float = 0.05
    magnitude: float = 5.0
    seed: Optional[int] = None


class ObservationPerturbationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, config: PerturbationConfig):
        super().__init__(env)
        self.config = config
        self._random = np.random.default_rng(config.seed)

        if config.enabled:
            logger.info(f"Observation perturbation enabled with "
                        f"frequency={config.frequency}, "
                        f"magnitude={config.magnitude}")
        else:
            logger.info("Observation perturbation disabled")

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)

        if self.config.enabled:
            obs, info = self._maybe_perturb_observation(obs, info)

        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.config.enabled:
            obs, info = self._maybe_perturb_observation(obs, info)

        return obs, reward, terminated, truncated, info

    def _maybe_perturb_observation(self, obs: Any, info: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        if self._random.random() < self.config.frequency:
            # Generate perturbation
            if isinstance(obs, np.ndarray):
                # For array observations, perturb random elements
                perturb_idx = self._random.integers(0, obs.shape[0])
                perturbation = self._random.normal(0, self.config.magnitude, size=1)
                perturbed_obs = obs.copy()
                perturbed_obs[perturb_idx] += perturbation

                logger.debug(f"Perturbed observation at index {perturb_idx} by {perturbation}")
                info["perturbed"] = True
                return perturbed_obs, info
            else:
                logger.warning(f"Unsupported observation type: {type(obs)}")
        return obs, info

def wrap_env_with_perturbation(env: gym.Env, config: PerturbationConfig) -> gym.Env:
    if config.enabled:
        return ObservationPerturbationWrapper(env, config)
    return env
