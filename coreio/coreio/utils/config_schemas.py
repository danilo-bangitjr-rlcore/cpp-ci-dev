from datetime import timedelta

from lib_config.config import config, list_
from pydantic import Field

from coreio.config import CoreIOConfig, TagConfigAdapter


@config(allow_extra=True, frozen=True)
class TagDBConfigAdapter:
    table_name: str = "sensors"

@config(allow_extra=True, frozen=True)
class AsyncEnvConfigAdapter:
    db: TagDBConfigAdapter = Field(default_factory=TagDBConfigAdapter)

@config(allow_extra=True)
class DBConfigAdapter:
    drivername: str = 'postgresql+psycopg2'
    username: str = 'postgres'
    password: str = 'password'
    ip: str = 'localhost'
    port: int = 5432
    db_name: str = 'postgres'
    schema: str = 'public'

@config(allow_extra=True, frozen=True)
class InfraConfigAdapter:
    db: DBConfigAdapter = Field(default_factory=DBConfigAdapter)

@config(allow_extra=True, frozen=True)
class PipelineConfigAdapter:
    tags: list[TagConfigAdapter] = list_()

@config(allow_extra=True, frozen=True)
class HeartbeatConfigAdapter:
    connection_id: str | None = ""
    heartbeat_node_id: str | None = ""

@config(allow_extra=True, frozen=True)
class InteractionConfigAdapter:
    heartbeat: HeartbeatConfigAdapter | None = Field(default_factory=HeartbeatConfigAdapter)
    action_period: timedelta | None = None

@config(allow_extra=True, frozen=True)
class MainConfigAdapter:
    coreio: CoreIOConfig = Field(default_factory=CoreIOConfig)
    env: AsyncEnvConfigAdapter = Field(default_factory=AsyncEnvConfigAdapter)
    infra: InfraConfigAdapter = Field(default_factory=InfraConfigAdapter)
    pipeline: PipelineConfigAdapter | None = None
    interaction: InteractionConfigAdapter | None = None
