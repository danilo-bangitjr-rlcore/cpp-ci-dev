from typing import Any
from dataclasses import field
from corerl.configs.config import MISSING, config


@config()
class EnvironmentConfig:
    type: str = 'gym.make'
    name: str = MISSING
    seed: int = MISSING
    discrete_control: bool = MISSING

    # gym environment init args and kwargs, ignored for deployment_async_env
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
