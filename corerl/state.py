from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable

if TYPE_CHECKING:
    from corerl.config import MainConfig
    from corerl.messages.event_bus import EventBus


@dataclass
class AppState:
    cfg: MainConfig
    evals: EvalsTable
    metrics: MetricsTable
    event_bus: EventBus
    agent_step: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))

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
        with open(path / 'state.pkl', 'rb') as f:
          state = pickle.load(f)

        self.agent_step = state.agent_step
        self.start_time = state.start_time
        return self
