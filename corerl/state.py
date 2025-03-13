from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from corerl.eval.evals import EvalTableProtocol
from corerl.eval.metrics import MetricsTableProtocol

if TYPE_CHECKING:
    from corerl.config import MainConfig
    from corerl.messages.event_bus import EventBus


@dataclass
class AppState:
    cfg: MainConfig
    evals: EvalTableProtocol
    metrics: MetricsTableProtocol
    event_bus: EventBus
    agent_step: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __getstate__(self):
      return {}

    def __setstate__(self, state: dict):
        ...
