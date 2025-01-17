from dataclasses import dataclass

from corerl.eval.writer import MetricsWriterProtocol


@dataclass
class AppState:
    metrics: MetricsWriterProtocol
    event_bus: None
