import uuid
from pydantic import BaseModel, Field

from corerl.utils.time import now_iso

from enum import StrEnum, auto

class IOEventType(StrEnum):
    # ------------
    # -- CoreIO --
    # ------------
    write_opcua_nodes = auto()

class IOEventTopic(StrEnum):
    # Topic filtering occurs using subscriber-side prefixing
    coreio = auto()

class IOEvent(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    time: str = Field(default_factory=now_iso)
    type: IOEventType
