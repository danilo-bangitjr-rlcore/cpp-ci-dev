from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

from corerl.configs.config import computed, config
from corerl.configs.loader import config_from_dict

if TYPE_CHECKING:
    from corerl.config import MainConfig

@config()
class DistractionWorldConfig:
    num_distractors : int = 100
    num_actions : int = 1
    seed: int = 0

    @computed('seed')
    @classmethod
    def _seed(cls, cfg: 'MainConfig'):
        return cfg.experiment.seed


class DistractionWorld(gym.Env):
    """
    Environment where there are a number of distractor states, sampled from a gaussian.
    The agents job is simply to ignore the sates and take an action of 0.5 every step.
    """
    def __init__(self, cfg: dict | DistractionWorldConfig | None = None):
        if isinstance(cfg, dict):
            cfg_or_err = config_from_dict(DistractionWorldConfig, cfg)
            assert isinstance(cfg_or_err, DistractionWorldConfig)
            cfg = cfg_or_err
        elif cfg is None:
            cfg = DistractionWorldConfig()

        self.num_distractors = cfg.num_distractors
        self.num_actions = cfg.num_actions

        self._random = np.random.default_rng(cfg.seed)
        self._obs_min = -np.ones(self.num_distractors)
        self._obs_max = np.ones(self.num_distractors)
        self.observation_space = gym.spaces.Box(self._obs_min, self._obs_max, dtype=np.float64)

        self._action_min = np.zeros(self.num_actions)
        self._action_max = np.ones(self.num_actions)
        self.action_space = gym.spaces.Box(self._action_min, self._action_max, dtype=np.float64)
        self._action_sp = np.ones(self.num_actions)*.5

        self.state = self._gen_state()
        self.last_action = np.zeros(self.num_actions)

    def seed(self, seed: int):
        self._random = np.random.default_rng(seed)

    def _gen_state(self) -> np.ndarray:
        state = np.random.random(self.num_distractors)*2 - 1
        state = np.clip(state, -1, 1)
        return state

    def _get_reward(self, action: np.ndarray) -> float:
        reward = float(-np.abs(self._action_sp - action).mean())
        return reward

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = self._get_reward(action)
        self.state = self._gen_state()
        return self.state, reward, False, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:

        self.state = self._gen_state()
        return self.state, {}

    def close(self):
        pass


gym.register(
    id='DistractionWorld-v0',
    entry_point='corerl.environment.distraction_world:DistractionWorld'
)
