from dataclasses import dataclass

from corerl.eval.writer import MetricsWriter


@dataclass
class AppState:
    metrics: MetricsWriter
    event_bus: None
