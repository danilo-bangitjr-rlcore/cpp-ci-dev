from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from corerl.eval.evals import EvalTableProtocol
from corerl.eval.metrics import MetricsTableProtocol
from corerl.messages.event_bus import EventBus

if TYPE_CHECKING:
    from corerl.config import MainConfig


@dataclass
class AppState:
    cfg: MainConfig
    evals: EvalTableProtocol
    metrics: MetricsTableProtocol
    event_bus: EventBus
    agent_step: int = 0

    def __getstate__(self):
      return {}

    def __setstate__(self, state: dict):
        ...
