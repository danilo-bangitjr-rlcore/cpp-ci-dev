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
    app_time: AppTime = field(default_factory=_default_app_time)
    stop_event: threading.Event = field(default_factory=threading.Event)

    @property
    def agent_step(self) -> int:
        return self.app_time.agent_step

    @property
    def start_time(self) -> datetime:
        return self.app_time.start_time

    def __getstate__(self):
        return {
            'app_time': self.app_time.__getstate__(),
        }

    def __setstate__(self, state: dict):
        self.app_time.__setstate__(state['app_time'])

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'state.pkl', 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: Path):
        try:
            with open(path / 'state.pkl', 'rb') as f:
                state = pickle.load(f)

            self.app_time.agent_step = state.agent_step
            self.app_time.start_time = state.start_time
        except Exception:
            logger.exception('Failed to load app state from checkpoint. Reinitializing...')

        return self
