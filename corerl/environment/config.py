from corerl.configs.config import MISSING, config


@config()
class EnvironmentConfig:
    type: str = 'gym.make'
    name: str = MISSING
    seed: int = MISSING
    discrete_control: bool = MISSING

    render_mode: str | None = None
