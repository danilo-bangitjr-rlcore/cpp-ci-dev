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
    step = auto()
    step_get_obs = auto()
    step_agent_update = auto()
    step_emit_action = auto()
    agent_step = auto()

    # -----------
    # -- Agent --
    # -----------
    agent_get_action = auto()
    agent_heartbeat = auto()
    agent_load = auto()
    agent_save = auto()
    agent_update_actor = auto()
    agent_update_sampler = auto()
    agent_update_buffer = auto()
    agent_update_critic = auto()

    # --------------
    # -- Pipeline --
    # --------------
    red_zone_violation = auto()
    yellow_zone_violation = auto()

    # ----------
    # -- Coms --
    # ----------
    ping_setpoints = auto()

    # -----------
    # -- Debug --
    # -----------
    toggle_event_logging = auto()

class EventTopic(StrEnum):
    # Topic filtering occurs using subscriber-side prefixing
    corerl = auto()
    corerl_scheduler = auto()
    corerl_cli = auto()
    debug_app = auto()


class Event(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    time: str = Field(default_factory=now_iso)
    type: EventType


# ---------------------
# -- Utility methods --
# ---------------------
def maybe_parse_event(msg: str | bytes) -> Event | None:
    try:
        return Event.model_validate_json(msg)
    except ValidationError:
        logger.exception('Failed to parse message')
        return None
