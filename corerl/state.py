from __future__ import annotations

import logging
import pickle
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.messages.events import Event, EventTopic, EventType

if TYPE_CHECKING:
    from corerl.config import MainConfig

logger = logging.getLogger(__name__)


type Callback = Callable[[Event], Any]
class IEventBus(Protocol):
    def start(self): ...
    def cleanup(self): ...
    def emit_event(self, event: Event | EventType, topic: EventTopic = EventTopic.debug_app): ...
    def attach_callback(self, event_type: EventType, cb: Callback): ...
    def attach_callbacks(self, cbs: dict[EventType, Callback]): ...


@dataclass
class AppState[
    EventBus: IEventBus
]:
    cfg: MainConfig
    evals: EvalsTable
    metrics: MetricsTable
    event_bus: EventBus
    agent_step: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    stop_event: threading.Event = field(default_factory=threading.Event)

    def __getstate__(self):
        return {
            'agent_step': self.agent_step,
            'start_time': self.start_time,
        }

    def __setstate__(self, state: dict):
        self.agent_step = state['agent_step']
        self.start_time = state['start_time']

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'state.pkl', 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: Path) -> AppState:
        try:
            with open(path / 'state.pkl', 'rb') as f:
                state = pickle.load(f)

            self.agent_step = state.agent_step
            self.start_time = state.start_time
        except Exception:
            logger.exception('Failed to load app state from checkpoint. Reinitializing...')

        return self
