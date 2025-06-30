from datetime import timedelta

from lib_config.config import MISSING, config, list_
from lib_defs.config_defs.tag_config import TagType
from pydantic import Field

from coreio.config import CoreIOConfig


@config(allow_extra=True, frozen=True)
class TagDBConfigAdapter:
    table_name: str = "sensors"

@config(allow_extra=True, frozen=True)
class AsyncEnvConfigAdapter:
    db: TagDBConfigAdapter = Field(default_factory=TagDBConfigAdapter)

@config(allow_extra=True, frozen=True)
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
class TagConfigAdapter:
    type: TagType = TagType.default
    name: str = MISSING
    connection_id: str | None = None
    node_identifier: str | None = None

@config(allow_extra=True, frozen=True)
class PipelineConfigAdapter:
    tags: list[TagConfigAdapter] = list_()

@config(allow_extra=True, frozen=True)
class HeartbeatConfigAdapter:
    connection_id: str | None = ""
    heartbeat_node_id: str | None = ""

@config(allow_extra=True, frozen=True)
class InteractionConfigAdapter:
    heartbeat: HeartbeatConfigAdapter = Field(default_factory=HeartbeatConfigAdapter)
    obs_period: timedelta = MISSING
    action_period: timedelta = MISSING

@config(allow_extra=True, frozen=True)
class MainConfigAdapter:
    coreio: CoreIOConfig = Field(default_factory=CoreIOConfig)
    pipeline: PipelineConfigAdapter = Field(default_factory=PipelineConfigAdapter)
    interaction: InteractionConfigAdapter = Field(default_factory=InteractionConfigAdapter)
    env: AsyncEnvConfigAdapter = Field(default_factory=AsyncEnvConfigAdapter)
    infra: InfraConfigAdapter = Field(default_factory=InfraConfigAdapter)
