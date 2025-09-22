from datetime import timedelta

from lib_config.config import config, list_
from pydantic import Field

from coreio.config import CoreIOConfig, TagConfigAdapter


@config(allow_extra=True, frozen=True)
class TagDBConfigAdapter:
    table_name: str

@config(allow_extra=True, frozen=True)
class AsyncEnvConfigAdapter:
    db: TagDBConfigAdapter

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
    env: AsyncEnvConfigAdapter
    coreio: CoreIOConfig = Field(default_factory=CoreIOConfig)
    pipeline: PipelineConfigAdapter | None = Field(default_factory=PipelineConfigAdapter)
    interaction: InteractionConfigAdapter | None = Field(default_factory=InteractionConfigAdapter)
    infra: InfraConfigAdapter = Field(default_factory=InfraConfigAdapter)
