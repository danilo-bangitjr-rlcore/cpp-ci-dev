from datetime import timedelta
from typing import Any, Literal

import pandas as pd
from pydantic import Field

from corerl.configs.config import MISSING, config
from corerl.data_pipeline.db.data_reader import TagDBConfig
from corerl.environment.config import EnvironmentConfig


@config()
class BaseAsyncEnvConfig(EnvironmentConfig):
    obs_period: timedelta = MISSING
    update_period: timedelta = MISSING
    action_period: timedelta = MISSING
    setpoint_ping_period: timedelta | None = None

@config()
class OPCEnvConfig(BaseAsyncEnvConfig):
    opc_conn_url: str = MISSING
    opc_ns: int = MISSING  # OPC node namespace, this is almost always going to be `2`
    client_cert_path: str | None = None
    client_private_key_path: str | None = None
    server_cert_path: str | None = None

@config()
class TSDBEnvConfig(BaseAsyncEnvConfig):
    db: TagDBConfig = MISSING

@config()
class GymEnvConfig:
    gym_name: str = MISSING
    init_type: Literal["gym.make", "custom"] | None = "gym.make"
    seed: int | None = None

    # gym environment init args and kwargs, ignored for deployment_async_env
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class AsyncEnv:
    obs_period: timedelta
    update_period: timedelta
    action_period: timedelta
    action_tolerance: timedelta

    def emit_action(self, action: pd.DataFrame) -> None: ...

    def get_latest_obs(self) -> pd.DataFrame: ...

    def cleanup(self) -> None:
        return

    def get_cfg(self) -> BaseAsyncEnvConfig: ...
