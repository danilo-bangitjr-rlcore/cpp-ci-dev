from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

from lib_config.config import MISSING, computed, config
from pydantic import Field
from rl_env.factory import EnvConfig

from corerl.configs.data_pipeline.db.data_writer import TagDBConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class AsyncEnvConfig:
    name: Literal["dep_async_env", "sim_async_env"] = "dep_async_env"
    obs_period: timedelta = MISSING
    coreio_origin: str = MISSING
    db: TagDBConfig = Field(default_factory=TagDBConfig)
    gym: GymEnvConfig | None = None

    @computed('coreio_origin')
    @classmethod
    def _coreio_origin(cls, cfg: MainConfig):
        return cfg.coreio.coreio_origin

    @computed('obs_period')
    @classmethod
    def _obs_period(cls, cfg: MainConfig):
        return cfg.interaction.obs_period

@config()
class WrapperConfig:
    name: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

@config()
class GymEnvConfig:
    init_type: Literal["gym.make", "custom", "model"] | None = "custom"
    wrapper: WrapperConfig = Field(default_factory=WrapperConfig)
    seed: int = MISSING

    # env config for custom gym environments
    env_config: EnvConfig = MISSING

    # perturb config for perturbation resilience tests
    perturb_config: Any | None = None
    @computed('seed')
    @classmethod
    def _seed(cls, cfg: MainConfig):
        return cfg.seed
