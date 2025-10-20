from lib_config.config import config
from pydantic import Field


@config()
class DBConfig:
    """
    Kind: optional external

    Default configurations for our timeseries
    database. Should generally be owned by RLCore
    during setup, however in more extreme circumstances
    this may be owned by the user on an internal network.
    """
    drivername: str = 'postgresql+psycopg2'
    username: str = 'postgres'
    password: str = 'password'
    ip: str = 'localhost'
    port: int = 5432
    db_name: str = 'postgres'
    schema: str = 'public'


@config()
class InfraConfig:
    """
    Kind: optional external

    Infrastructure configuration for wiring the agent
    into the external system.
    """
    db: DBConfig = Field(default_factory=DBConfig)


@config()
class FeatureFlags:
    """
    Kind: internal

    Flags to enable/disable new features in the agent.
    These should default to False and the date that they
    are added should be recorded.

    Feature flags should generally be for internal use only.
    Their impact on the customer cannot be easily guaranteed
    or communicated.

    See documentation:
    https://docs.google.com/document/d/1Inm7dMHIRvIGvM7KByrRhxHsV7uCIZSNsddPTrqUcOU/edit?tab=t.4238yb3saoju
    """
    # 2025-02-01
    ensemble: int = 2

    # 2025-04-29
    recency_bias_buffer: bool = False

    # 2025-05-14
    regenerative_optimism: bool = False

    # 2025-05-26
    normalize_return: bool = False

    # 2025-06-11
    nominal_setpoint_bias: bool = True

    # 2025-06-27
    higher_critic_lr: bool = True

    # 2025-06-27
    mu_sigma_multipliers: bool = False

    # 2025-07-24
    wide_metrics: bool = False

    # 2025-10-02
    all_layer_norm: bool = False
