from dataclasses import field
from typing import Any

import numpy as np
import pandas as pd

from corerl.configs.config import MISSING, config


@config()
class BaseAsyncEnvConfig:
    name: str = MISSING
    gym_name: str = MISSING
    seed: int = 0
    discrete_control: bool = False

    # gym environment init args and kwargs, ignored for deployment_async_env
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)


class AsyncEnv:
    def emit_action(self, action: np.ndarray) -> None: ...

    def get_latest_obs(self) -> pd.DataFrame: ...
