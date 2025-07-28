import uuid
from enum import StrEnum

from lib_utils.time import now_iso
from pydantic import BaseModel, Field


class BaseEventType(StrEnum):
    ...

class BaseEventTopic(StrEnum):
    ...

class BaseEvent[EventTypeClass: BaseEventType](BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    time: str = Field(default_factory=now_iso)
    type: EventTypeClass
