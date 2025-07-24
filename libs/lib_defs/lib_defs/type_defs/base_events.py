import uuid
from enum import StrEnum
from typing import Generic, TypeVar

from lib_utils.time import now_iso
from pydantic import BaseModel, Field


class BaseEventType(StrEnum):
    ...

class BaseEventTopic(StrEnum):
    ...

EventTypeClass = TypeVar('EventTypeClass', bound=BaseEventType)
EventTopicClass = TypeVar('EventTopicClass', bound=BaseEventTopic)

class BaseEvent(BaseModel, Generic[EventTypeClass]): # noqa: UP046
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    time: str = Field(default_factory=now_iso)
    type: EventTypeClass

EventClass = TypeVar("EventClass", bound=BaseEvent)
