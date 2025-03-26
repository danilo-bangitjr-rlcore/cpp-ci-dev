from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from pydantic import Field

from corerl.configs.config import MISSING, computed, config
from corerl.data_pipeline.db.data_reader import TagDBConfig
from corerl.environment.config import EnvironmentConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig


# -------------
# -- Configs --
# -------------
@config()
class BaseAsyncEnvConfig(EnvironmentConfig):
    obs_period: timedelta = MISSING
    update_period: timedelta = MISSING
    action_period: timedelta = MISSING
    setpoint_ping_period: timedelta | None = None

    @computed('obs_period')
    @classmethod
    def _obs_period(cls, cfg: MainConfig):
        return cfg.interaction.obs_period

    @computed('action_period')
    @classmethod
    def _action_period(cls, cfg: MainConfig):
        return cfg.interaction.action_period

    @computed('update_period')
    @classmethod
    def _update_period(cls, cfg: MainConfig):
        return cfg.interaction.obs_period


@config()
class TSDBEnvConfig(BaseAsyncEnvConfig):
    db: TagDBConfig = Field(default_factory=TagDBConfig)


@config()
class GymEnvConfig:
    gym_name: str = MISSING
    init_type: Literal["gym.make", "custom", "model"] | None = "gym.make"
    seed: int = MISSING

    # gym environment init args and kwargs, ignored for deployment_async_env
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    # env config for custom gym environments
    env_config: Any | None = None

    @computed('seed')
    @classmethod
    def _seed(cls, cfg: 'MainConfig'):
        return cfg.experiment.seed


@config()
class DepAsyncEnvConfig(TSDBEnvConfig):
    """Configuration for the deployment async environment.

    Attributes:
        name                Discriminator for configuration dispatcher.
        action_tolerance    Computed from interaction obs_period.
        coreio_origin       Endpoint for CoreIO Thin Client web service. Defaults to docker-compose up expected value.
    """

    name: Literal["dep_async_env"] = "dep_async_env"
    action_tolerance: timedelta = MISSING
    coreio_origin: str = "http://coreio:2222"

    @computed('action_tolerance')
    @classmethod
    def _action_tolerance(cls, cfg: MainConfig):
        return cfg.interaction.obs_period


@config()
class SimAsyncEnvConfig(GymEnvConfig, BaseAsyncEnvConfig):
    name: Literal["sim_async_env"] = "sim_async_env"

# ---------------
# -- Interface --
# ---------------
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
