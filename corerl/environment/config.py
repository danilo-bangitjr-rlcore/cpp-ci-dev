from corerl.configs.config import MISSING, config


@config()
class EnvironmentConfig:
    name: str = MISSING
    seed: int | None = None
    discrete_control: bool = MISSING

