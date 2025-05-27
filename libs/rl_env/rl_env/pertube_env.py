import logging
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PerturbationConfig():
    name: str = "Perturbed-v0"
    frequency: float = 0.05
    magnitude: float = 5.0
    seed: int = 0

class ObservationPerturbationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, cfg: PerturbationConfig):
        super().__init__(env)
        self.env = env
        self.config = cfg
        self._random = np.random.default_rng(cfg.seed)

        logger.info(f"Observation perturbation enabled with "
                        f"frequency={cfg.frequency}, "
                        f"magnitude={cfg.magnitude}")

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)

        obs, info = self._maybe_perturb_observation(obs, info)

        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

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


