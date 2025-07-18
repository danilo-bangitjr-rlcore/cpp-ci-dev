import uuid
from enum import StrEnum
from typing import Generic, TypeVar

from lib_utils.time import now_iso
from pydantic import BaseModel, Field


class BaseEventType(StrEnum):
    ...

class BaseEventTopic(StrEnum):
    ...

EventTypeVar = TypeVar('EventTypeVar', bound=BaseEventType)

class BaseEvent(BaseModel, Generic[EventTypeVar]): # noqa: UP046
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    time: str = Field(default_factory=now_iso)
    type: EventTypeVar
