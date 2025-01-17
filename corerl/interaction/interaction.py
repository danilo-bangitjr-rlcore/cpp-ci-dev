import logging
from typing import Protocol

from corerl.messages.events import Event, EventType

logger = logging.getLogger(__file__)


class Interaction(Protocol):
    def step(self) -> None: ...

    def step_event(self, event: Event):
        match event.type:
            case EventType.step:
                self.step()
            case _:
                logger.info(f"Got unexpected event: {event}")
