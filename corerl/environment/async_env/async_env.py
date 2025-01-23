from dataclasses import field
from datetime import timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd

from corerl.configs.config import MISSING, config
from corerl.data_pipeline.db.data_reader import TagDBConfig
from corerl.environment.config import EnvironmentConfig


@config()
class BaseAsyncEnvConfig(EnvironmentConfig):
    obs_period: timedelta = MISSING
    update_period: timedelta = MISSING
    action_period: timedelta = MISSING
    setpoint_ping_period: timedelta = timedelta(seconds=5)

@config()
class OPCEnvConfig(BaseAsyncEnvConfig):
    opc_conn_url: str = MISSING
    opc_ns: int = MISSING  # OPC node namespace, this is almost always going to be `2`

@config()
class TSDBEnvConfig(BaseAsyncEnvConfig):
    db: TagDBConfig = MISSING

@config()
class GymEnvConfig(BaseAsyncEnvConfig):
    gym_name: str = MISSING
    init_type: Literal["gym.make", "custom"] | None = "gym.make"

    # gym environment init args and kwargs, ignored for deployment_async_env
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)


class AsyncEnv:
    obs_period: timedelta
    update_period: timedelta
    action_period: timedelta
    action_tolerance: timedelta

    def emit_action(self, action: np.ndarray) -> None: ...

    def get_latest_obs(self) -> pd.DataFrame: ...

    def cleanup(self) -> None:
        return
