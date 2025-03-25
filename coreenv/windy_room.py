from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

from corerl.configs.config import computed, config
from corerl.configs.loader import config_from_dict

if TYPE_CHECKING:
    from corerl.config import MainConfig

@config()
class WindyRoomConfig:
    seed: int = 0
    initial_zone_low: float = 0.45
    initial_zone_high: float = 0.55

    initial_wind_direction = np.pi
    wind_magnitude: float  = 0.01
    wind_direction_delta: float = (2*np.pi)/1000 # 1000 steps to complete a full circle
    action_magnitude: float = 0.02

    @computed('seed')
    @classmethod
    def _seed(cls, cfg: 'MainConfig'):
        return cfg.experiment.seed

STATE_DIM = 2
BOUNDS_LOW = 0
BOUNDS_HIGH = 1

class WindyRoom(gym.Env):
    """
    Environment where the agent exists in a room where a "wind"
    is blowing in a direction that is slowly changing over time.
    The agent starts in a random position in the room within an initial zone
    and can move in any direction.
    """
    def __init__(self, cfg: WindyRoomConfig | None = None):
        if cfg is None:
            cfg = WindyRoomConfig()

        self._cfg = cfg
        self._random = np.random.default_rng(cfg.seed)
        self._obs_min = np.ones(STATE_DIM) * BOUNDS_LOW
        self._obs_max = np.ones(STATE_DIM) * BOUNDS_HIGH
        self.observation_space = gym.spaces.Box(self._obs_min, self._obs_max, dtype=np.float64)

        self._action_min = -np.ones(STATE_DIM)
        self._action_max = np.ones(STATE_DIM)
        self.action_space = gym.spaces.Box(self._action_min, self._action_max, dtype=np.float64)

        self.initial_zone_low = np.ones(STATE_DIM) * cfg.initial_zone_low
        self.initial_zone_high = np.ones(STATE_DIM) * cfg.initial_zone_high

        self.wind_direction = cfg.initial_wind_direction

    def seed(self, seed: int):
        self._random = np.random.default_rng(seed)


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        xy = self.state[:STATE_DIM]
        # increment position components of the state
        xy = xy + self._cfg.action_magnitude*action
        wind_delta = np.array([np.cos(self.wind_direction), np.sin(self.wind_direction)])*self._cfg.wind_magnitude
        xy = xy + wind_delta
        xy = np.clip(xy, BOUNDS_LOW, BOUNDS_HIGH)

        # adjust wind direction
        self.wind_direction = self.wind_direction + self._cfg.wind_direction_delta
        if self.wind_direction > 2*np.pi:
            self.wind_direction = self.wind_direction - 2*np.pi

        self.state = np.concatenate([xy, [self.wind_direction]])
        reward = 0.
        return self.state, reward, False, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        self.num_steps = 0
        self.wind_direction = self._cfg.initial_wind_direction
        xy = np.random.random(STATE_DIM) * (self.initial_zone_high - self.initial_zone_low) + self.initial_zone_low
        self.state = np.concatenate([xy, [self.wind_direction]])
        return self.state, {}

    def close(self):
        pass


gym.register(
    id='WindyRoom-v0',
    entry_point='corerl.environment.windy_room:WindyRoom',
)
