from __future__ import annotations

import logging
import pickle
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from corerl.eval.evals.base import EvalsWriterProtocol
from corerl.eval.metrics.base import MetricsWriterProtocol
from corerl.messages.event_bus import DummyEventBus, EventBus
from corerl.utils.app_time import AppTime

if TYPE_CHECKING:
    from corerl.config import MainConfig


logger = logging.getLogger(__name__)


def _default_app_time() -> AppTime:
    """Create default AppTime for normal operation (non-demo mode)."""
    return AppTime(
        is_demo=False,
        start_time=datetime.now(UTC),
    )


@dataclass
class AppState[AppEventBus: (EventBus, DummyEventBus), AppMainConfig: MainConfig]:
    cfg: AppMainConfig
    evals: EvalsWriterProtocol
    metrics: MetricsWriterProtocol
    event_bus: AppEventBus
    agent_step: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    app_time: AppTime = field(default_factory=_default_app_time)
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

    def load(self, path: Path):
        try:
            with open(path / 'state.pkl', 'rb') as f:
                state = pickle.load(f)

            self.agent_step = state.agent_step
            self.start_time = state.start_time
        except Exception:
            logger.exception('Failed to load app state from checkpoint. Reinitializing...')

        return self
