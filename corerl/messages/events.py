import uuid
from enum import StrEnum, auto
from pydantic import BaseModel, Field

from corerl.utils.time import now_iso


class EventType(StrEnum):
    # ---------------
    # -- Lifecycle --
    # ---------------
    subscribe = auto()

    # -----------
    # -- Agent --
    # -----------
    agent_get_action = auto()
    agent_heartbeat = auto()
    agent_load = auto()
    agent_save = auto()
    agent_update_actor = auto()
    agent_update_buffer = auto()
    agent_update_critic = auto()



class Event(BaseModel):
    id: uuid.UUID = Field(..., default_factory=uuid.uuid4)
    time: str = Field(..., default_factory=now_iso)
    type: EventType


class SubscribeEvent(Event):
    type: EventType = EventType.subscribe
    subscribe_to: EventType
