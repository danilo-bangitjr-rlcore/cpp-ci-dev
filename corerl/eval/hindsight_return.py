
import numpy as np
import pandas as pd

from corerl.configs.config import config
from corerl.state import AppState


@config()
class HindsightReturnConfig:
    enabled: bool = True

class HindsightReturnEval:
    def __init__(self, cfg: HindsightReturnConfig, app_state: AppState):
        self._app_state = app_state
        self.enabled = cfg.enabled

        self.gamma = app_state.cfg.agent.gamma
        self.trace = None

    def execute(self, rewards: pd.DataFrame):
        if not self.enabled:
            return

        reward_np = rewards['reward'].values

        for r in reward_np:
            # first reward is always nan
            if np.isnan(r):
                continue

            if self.trace is None:
                self.trace = r
            else:
                self.trace = self.trace * self.gamma + r

        if self.trace is not None:
            self._app_state.metrics.write(self._app_state.agent_step, 'hindsight_return', self.trace)

    def reset(self):
        self.trace = None
