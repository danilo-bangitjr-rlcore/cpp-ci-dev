import uuid
from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict, Field

from coreio.utils.opc_communication import OPCUANodeWriteValue
from corerl.utils.time import now_iso


class IOEventType(StrEnum):
    # ------------
    # -- CoreIO --
    # ------------
    write_opcua_nodes = auto()
    read_opcua_nodes = auto()
    exit_io = auto()

class IOEventTopic(StrEnum):
    # Topic filtering occurs using subscriber-side prefixing
    coreio = auto()

class IOEvent(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    time: str = Field(default_factory=now_iso)
    type: IOEventType
    data: dict[str, list[OPCUANodeWriteValue]] = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)

