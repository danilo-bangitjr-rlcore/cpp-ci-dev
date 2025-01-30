from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from corerl.eval.evals import EvalWriterProtocol
from corerl.eval.metrics import MetricsWriterProtocol
from corerl.messages.event_bus import EventBus

if TYPE_CHECKING:
    from corerl.config import MainConfig


@dataclass
class AppState:
    cfg: MainConfig
    evals: EvalWriterProtocol
    metrics: MetricsWriterProtocol
    event_bus: EventBus
    agent_step: int = 0
