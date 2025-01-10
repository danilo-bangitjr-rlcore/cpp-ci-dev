from corerl.configs.config import MISSING, config


@config()
class EnvironmentConfig:
    name: str = MISSING
    seed: int = MISSING
    discrete_control: bool = MISSING

