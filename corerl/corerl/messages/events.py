import logging
import uuid
from enum import auto

from lib_defs.type_defs.base_events import BaseEvent, BaseEventTopic, BaseEventType
from lib_utils.time import now_iso
from pydantic import Field, ValidationError

logger = logging.getLogger(__name__)


class RLEventType(BaseEventType):
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
    action_period_reset = auto()

    # ----------
    # -- Coms --
    # ----------
    ping_setpoints = auto()
    flush_buffers = auto()

class RLEventTopic(BaseEventTopic):
    # Topic filtering occurs using subscriber-side prefixing
    corerl = auto()
    corerl_scheduler = auto()
    corerl_cli = auto()
    debug_app = auto()


class RLEvent(BaseEvent[RLEventType]):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    time: str = Field(default_factory=now_iso)
    type: RLEventType


# ---------------------
# -- Utility methods --
# ---------------------
def maybe_parse_event(msg: str | bytes) -> RLEvent | None:
    try:
        return RLEvent.model_validate_json(msg)
    except ValidationError:
        logger.exception('Failed to parse message')
        return None
