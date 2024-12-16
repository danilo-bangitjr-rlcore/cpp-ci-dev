import numpy as np
import pandas as pd
import gymnasium as gym
from datetime import datetime, timedelta
from corerl.configs.config import config, MISSING
from dataclasses import dataclass
from corerl.data_pipeline.tag_config import TagConfig
from corerl.environment.async_env.async_env import AsyncEnv
from corerl.utils.gym import space_bounds


@config()
class SimAsyncEnvConfig:
    name: str = MISSING
    discrete_control: bool = False
    seed: int = 0


@dataclass
class StepData:
    obs: np.ndarray
    r: float
    a: np.ndarray
    trunc: bool
    term: bool


class SimAsyncEnv(AsyncEnv):
    def __init__(self, cfg: SimAsyncEnvConfig, tags: list[TagConfig]):
        self._env = gym.make(cfg.name)
        self._cfg = cfg

        shape = self._env.observation_space.shape
        assert shape is not None
        assert len(shape) == 1, 'Cannot handle environments with non-vector observations'

        self._action_bounds = space_bounds(self._env.action_space)

        # len(tags) should be the observation length
        # + one tag for action, reward, trunc, term
        assert shape[0] + 4 == len(tags), 'Received an unexpected number of tag configs'

        self._tag_names = [
            tag.name for tag in tags
            if tag.name not in {'reward', 'action', 'trunc', 'term'}
        ]

        self.clock = datetime(1984, 1, 1)
        self._clock_inc = timedelta(minutes=5)

        self._action: np.ndarray | None = None
        self._last_step: StepData | None = None


    # ------------------
    # -- AsyncEnv API --
    # ------------------
    def emit_action(self, action: np.ndarray) -> None:
        lo, hi = self._action_bounds
        scale = hi - lo
        bias = lo
        self._action = scale * action + bias


    def get_latest_obs(self) -> pd.DataFrame:
        self.clock += self._clock_inc

        if self._action is None or self._last_step is None or (self._last_step.term or self._last_step.trunc):
            step = self._reset()

        else:
            step = self._step(self._action)

        return self._obs_as_df(step)

    # -------------------------
    # -- Gym API Translation --
    # -------------------------
    def _reset(self):
        obs, *_ = self._env.reset()

        self._last_step = StepData(
            obs=obs,
            r=np.nan,
            a=np.array([np.nan]),
            trunc=False,
            term=False,
        )
        return self._last_step

    def _step(self, action: np.ndarray):
        assert len(action) == 1

        obs, r, term, trunc, _ = self._env.step(action)
        self._last_step = StepData(
            obs=obs,
            r=float(r),
            a=action,
            trunc=trunc,
            term=term,
        )
        return self._last_step


    def _obs_as_df(self, step: StepData):
        obs_data = {
            tag: val
            for tag, val in zip(self._tag_names, step.obs, strict=True)
        }

        non_obs_data = {
            'reward': step.r,
            'action': step.a,
            'trunc': step.trunc,
            'term': step.term,
        }

        idx = pd.DatetimeIndex([self.clock])
        df = pd.DataFrame(obs_data | non_obs_data, index=idx)
        return df
