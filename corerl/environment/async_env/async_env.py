import numpy as np
import pandas as pd

from corerl.configs.config import MISSING, config
from corerl.environment.config import EnvironmentConfig


@config()
class BaseAsyncEnvConfig(EnvironmentConfig):
    gym_name: str = MISSING

class AsyncEnv:
    def emit_action(self, action: np.ndarray) -> None: ...

    def get_latest_obs(self) -> pd.DataFrame: ...

    def cleanup(self) -> None:
        return
