import logging
from datetime import timedelta
from typing import Protocol

from corerl.configs.config import MISSING, config
from corerl.messages.events import Event, EventType

logger = logging.getLogger(__file__)


@config()
class BaseInteractionConfig:
    obs_period: timedelta = MISSING
    action_period: timedelta = MISSING


class Interaction(Protocol):
    def step(self) -> None: ...

    def step_event(self, event: Event):
        match event.type:
            case EventType.step:
                self.step()
            case _:
                logger.debug(f"Got unexpected event: {event}")
