from dataclasses import field
from typing import Any
from corerl.configs.config import MISSING, config

@config()
class EnvironmentConfig:
    type: str = 'gym.make'
    name: str = MISSING
    seed: int = MISSING
    discrete_control: bool = MISSING

    render_mode: str | None = None
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
