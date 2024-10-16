import logging
import uuid
from enum import StrEnum, auto
from pydantic import BaseModel, Field, ValidationError

from corerl.utils.time import now_iso

logger = logging.getLogger(__name__)



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


# ---------------------
# -- Utility methods --
# ---------------------
def maybe_parse_event(msg: str | bytes) -> Event | None:
    try:
        return Event.model_validate_json(msg)
    except ValidationError:
        logger.exception('Failed to parse websocket message')
        return None
