from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from pydantic import Field

from corerl.configs.config import MISSING, computed, config
from corerl.data_pipeline.db.data_reader import TagDBConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig


# -------------
# -- Configs --
# -------------
@config()
class AsyncEnvConfig:
    name: Literal["dep_async_env", "sim_async_env"] = "dep_async_env"
    obs_period: timedelta = MISSING
    coreio_origin: str = "http://coreio:2222"
    db: TagDBConfig = Field(default_factory=TagDBConfig)
    gym: GymEnvConfig | None = None

    @computed('obs_period')
    @classmethod
    def _obs_period(cls, cfg: MainConfig):
        return cfg.interaction.obs_period


@config()
class GymEnvConfig:
    gym_name: str = MISSING
    init_type: Literal["gym.make", "custom", "model"] | None = "custom"
    seed: int = MISSING

    # gym environment init args and kwargs, ignored for deployment_async_env
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    # env config for custom gym environments
    env_config: Any | None = None

    @computed('seed')
    @classmethod
    def _seed(cls, cfg: 'MainConfig'):
        return cfg.seed

# ---------------
# -- Interface --
# ---------------
class AsyncEnv:
    def emit_action(self, action: pd.DataFrame) -> None: ...

    def get_latest_obs(self) -> pd.DataFrame: ...

    def cleanup(self) -> None:
        return

    def get_cfg(self) -> AsyncEnvConfig: ...
