import logging
import uuid
from enum import StrEnum, auto

from lib_utils.time import now_iso
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class BaseEventType(StrEnum):
    ...


class BaseEventTopic(StrEnum):
    ...


class BaseEvent[EventTypeClass: BaseEventType](BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    time: str = Field(default_factory=now_iso)
    type: EventTypeClass


class EventType(BaseEventType):
    # ---------------
    # -- Lifecycle --
    # ---------------
    service_started = auto()
    service_stopped = auto()
    service_error = auto()
    service_heartbeat = auto()

    # ---------------
    # -- CoreRL --
    # ---------------
    step = auto()
    step_get_obs = auto()
    step_agent_update = auto()
    step_emit_action = auto()
    agent_step = auto()
    agent_get_action = auto()
    agent_load = auto()
    agent_save = auto()
    agent_update_actor = auto()
    agent_update_sampler = auto()
    agent_update_buffer = auto()
    agent_update_critic = auto()
    red_zone_violation = auto()
    yellow_zone_violation = auto()
    action_period_reset = auto()
    ping_setpoints = auto()
    flush_buffers = auto()

    # ---------------
    # -- CoreIO --
    # ---------------
    write_to_opc = auto()
    read_from_opc = auto()
    exit_io = auto()


class EventTopic(BaseEventTopic):
    corerl = auto()
    corerl_scheduler = auto()
    corerl_cli = auto()
    coreio = auto()
    coreio_debug = auto()
    coredinator = auto()
    debug_app = auto()


class Event(BaseEvent[EventType]):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    time: str = Field(default_factory=now_iso)
    type: EventType


def maybe_parse_event(msg: str | bytes) -> Event | None:
    try:
        return Event.model_validate_json(msg)
    except ValidationError:
        logger.exception("Failed to parse event message")
        return None
